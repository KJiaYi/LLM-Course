import numpy as np
import time
import airsim
from uav_data_generator import generate_task_points
from uav_maddpg_model import SimpleMADDPG
from uav_task_assignment import assign_tasks
from uav_path_planning import get_uav_state, distributed_path_planning
from uav_communication_sim import communicate_uav_states


def uav_swarm_patrol():
    # 1. 初始化AirSim客户端
    client = airsim.MultirotorClient()
    client.confirmConnection()
    uav_names = ["UAV0", "UAV1", "UAV2"]

    # 2. 初始化无人机
    for uav in uav_names:
        client.enableApiControl(True, uav)
        client.armDisarm(True, uav)
        client.takeoffAsync(vehicle_name=uav).join()
        client.moveToZAsync(-20, 5, vehicle_name=uav).join()

    # 3. 任务点和分配（使用新的多任务分配）
    task_points = generate_task_points(10)
    uav_positions = [
        np.array([client.getMultirotorState(uav).kinematics_estimated.position.x_val,
                  client.getMultirotorState(uav).kinematics_estimated.position.y_val,
                  client.getMultirotorState(uav).kinematics_estimated.position.z_val])
        for uav in uav_names
    ]
    primary_tasks, all_assigned_tasks = assign_tasks(uav_positions, task_points)

    # 4. 初始化MARL模型
    maddpg = SimpleMADDPG(num_uavs=3, state_dim=8, action_dim=3)

    # 5. 巡检参数（增加最大步数）
    max_steps = 300  # 从200增加到300
    step = 0.4  # 步长略微减小
    covered_points = set()
    collision_count = 0
    start_time = time.time()

    # 6. 集群巡检主循环
    for step_idx in range(max_steps):
        # 6.1 获取所有无人机状态
        uav_states = []
        current_tasks = []
        for uav, tasks in zip(uav_names, all_assigned_tasks):
            state = get_uav_state(client, uav, task_points, tasks)
            uav_states.append(state)

        # 6.2 通信模拟
        comm_states = communicate_uav_states(uav_states, uav_names)

        # 6.3 分布式路径规划+避障
        actions = []
        current_tasks = []
        for uav, state, tasks in zip(uav_names, uav_states, all_assigned_tasks):
            action, current_task = distributed_path_planning(client, uav, maddpg, state, tasks, task_points)
            actions.append(action)
            current_tasks.append(current_task)

        # 6.4 执行动作
        for uav, action in zip(uav_names, actions):
            client.moveByVelocityAsync(
                action[0], action[1], action[2],
                duration=step, vehicle_name=uav
            ).join()

        # 6.5 检测碰撞（更灵敏）
        current_collision = False
        for uav in uav_names:
            collision_info = client.simGetCollisionInfo(vehicle_name=uav)
            if collision_info.has_collided:
                collision_count += 1
                current_collision = True
                # 更合理的碰撞后重置
                client.moveToPositionAsync(5 + int(uav[-1]) * 10, 5 + int(uav[-1]) * 10, -20, 5, vehicle_name=uav).join()

        # 6.6 检测任务点覆盖（调整阈值）
        for uav, state, task in zip(uav_names, uav_states, current_tasks):
            if task is not None and state[6] < 6:  # 从8米调整为6米
                covered_points.add(task)

        # 6.7 MARL模型训练
        next_uav_states = [get_uav_state(client, uav, task_points, tasks) for uav, tasks in
                           zip(uav_names, all_assigned_tasks)]
        rewards = maddpg.calculate_reward_optimized(uav_states, task_points, covered_points, current_collision, all_assigned_tasks)
        critic_loss, actor_loss = maddpg.train_step(uav_states, actions, rewards, next_uav_states)

        # 6.8 打印进度
        if step_idx % 10 == 0:
            coverage_rate = len(covered_points) / len(task_points) * 100
            print(
                f"Step {step_idx}: 覆盖率={coverage_rate:.1f}%, 碰撞次数={collision_count}, CriticLoss={critic_loss:.4f}")

        time.sleep(step)

        # 6.9 任务完成条件
        if len(covered_points) >= 0.8 * len(task_points):  # 达到80%即可提前结束
            print("达到目标覆盖率，提前结束！")
            break

    # 7. 结束巡检
    end_time = time.time()
    task_time = end_time - start_time
    for uav in uav_names:
        client.landAsync(vehicle_name=uav).join()
        client.armDisarm(False, uav)
        client.enableApiControl(False, uav)

    # 8. 评估指标计算
    coverage_efficiency = len(covered_points) / len(task_points)
    collision_rate = collision_count / max_steps
    print(f"\n巡检完成！")
    print(f"覆盖效率：{coverage_efficiency:.2f}")
    print(f"碰撞率：{collision_rate:.4f}")
    print(f"任务完成时间：{task_time:.2f}秒")


if __name__ == "__main__":
    uav_swarm_patrol()
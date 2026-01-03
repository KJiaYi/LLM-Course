import numpy as np


def assign_tasks(uav_positions, task_points, covered_points, alpha=0.8):
    """
    动态优先级任务分配：
    - 未覆盖任务权重更高（alpha=0.8）
    - 已覆盖任务权重降低，避免重复巡检
    """
    num_uavs = len(uav_positions)
    num_tasks = len(task_points)
    assigned_tasks = [None] * num_uavs
    remaining_tasks = list(range(num_tasks))

    # 为任务分配优先级（未覆盖任务优先级×alpha权重）
    task_priorities = np.ones(num_tasks)
    for t in covered_points:
        if t < num_tasks:  # 防止索引越界
            task_priorities[t] = 1 - alpha  # 已覆盖任务优先级降低

    # 基于位置距离和任务优先级分配
    for uav_id in range(num_uavs):
        if not remaining_tasks:
            break
        # 计算加权距离（距离×(1/优先级)，优先级越高，距离权重越低）
        dists = []
        for t in remaining_tasks:
            raw_dist = np.linalg.norm(uav_positions[uav_id] - task_points[t])
            weighted_dist = raw_dist * (1 / task_priorities[t])  # 优先级高的任务距离被"缩短"
            dists.append(weighted_dist)
        best_task_idx = np.argmin(dists)
        assigned_tasks[uav_id] = remaining_tasks[best_task_idx]
        del remaining_tasks[best_task_idx]

    # 剩余任务强制分配给最近的无人机（确保全覆盖）
    if remaining_tasks:
        for t in remaining_tasks:
            task_pos = task_points[t]
            uav_dists = [np.linalg.norm(pos - task_pos) for pos in uav_positions]
            nearest_uav = np.argmin(uav_dists)
            assigned_tasks[nearest_uav] = t  # 覆盖旧任务，优先处理新任务

    return assigned_tasks


# 测试任务分配（可选执行）
if __name__ == "__main__":
    uav_pos = np.array([[0, 0, -20], [5, 0, -20], [10, 0, -20]])
    from uav_data_generator import generate_task_points
    task_pts = generate_task_points(10)
    assigned = assign_tasks(uav_pos, task_pts, covered_points={0,1})  # 测试已覆盖任务的分配
    print("动态任务分配结果：", assigned)
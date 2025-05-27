import random

class RobotArm:
    """
    Represents a robot arm in the industrial simulation environment.
    """
    def __init__(self, id: str, task_list: list, target_types: dict, target_list: list, location: int, task_info_mapping: dict):
        """
        Initializes a RobotArm instance.

        Args:
            id: Unique identifier for the robot arm.
            task_list: List of task types the robot arm can perform.
            target_types: Mapping of target工件需求.
            target_list: List of supported target工件.
            location: Spatial location of the robot arm.
            task_info_mapping: Configuration mapping for tasks.
        """
        self.id: str = id
        self.task: str = None  # Current task type
        self.task_list: list = task_list
        self.target_types: dict = target_types
        self.target_list: list = target_list
        self.location: int = location
        self.state: int = 0  # 0: idle, 1: working, 2: fault, 3: offline
        self.is_occupied: bool = False
        self.work_output: int = -1  # 1: OK, 0: NG, -1: no output
        self.output_type: str = None # Output工件type
        self.task_time: int = 0  # Time to execute current task
        self.task_switch_time: int = 0 # Time to switch to a new task
        self.failure_rate: float = 0.01 # Example failure rate
        self.recovery_rate: float = 0.1 # Example recovery rate
        self.ng_rate: float = 0.05 # Example NG rate
        self.task_info_mapping: dict = task_info_mapping

        # Example task_info_mapping structure:
        # {
        #     "task_id_1": {"time": 10, "switch_time": 2, "output_type": "type_A"},
        #     "task_id_2": {"time": 15, "switch_time": 3, "output_type": "type_B"},
        # }

    def reset(self):
        """
        Resets the robot arm to its initial state.
        """
        self.task = None
        self.state = 0  # idle
        self.is_occupied = False
        self.work_output = -1
        self.output_type = None
        self.task_time = 0
        print(f"RobotArm {self.id} has been reset.")

    def execute_task_check(self) -> bool:
        """
        Checks if the current task can be executed (e.g., preconditions met).
        This is a placeholder and should be implemented with actual logic.
        """
        print(f"RobotArm {self.id}: Checking preconditions for task {self.task}.")
        # Basic check: is the arm idle and not occupied?
        if self.state == 0 and not self.is_occupied:
            # Simulate failure based on failure_rate
            if random.random() < self.failure_rate:
                self.state = 2 # Fault
                print(f"RobotArm {self.id} encountered a fault before starting task {self.task}.")
                return False
            return True
        elif self.state == 1:
            print(f"RobotArm {self.id} is already working.")
            return False
        elif self.state == 2:
            print(f"RobotArm {self.id} is in fault state.")
            return False
        elif self.state == 3:
            print(f"RobotArm {self.id} is offline.")
            return False
        elif self.is_occupied:
            print(f"RobotArm {self.id} is occupied and cannot start a new task.")
            return False
        return False

    def execute_task(self, task_id: str):
        """
        Executes the assigned task.
        This is a placeholder and should be implemented with actual task execution logic.
        """
        if self.task != task_id and task_id in self.task_list:
            self.switch_task(task_id) # Switch task if different and valid

        if self.task == task_id and self.execute_task_check():
            self.state = 1  # working
            self.is_occupied = True
            task_details = self.task_info_mapping.get(self.task, {})
            self.task_time = task_details.get("time", 10) # Default time if not in mapping
            self.output_type = task_details.get("output_type", None)

            print(f"RobotArm {self.id}: Executing task {self.task} for {self.task_time} units of time.")
            # Simulate task completion and output
            # In a real scenario, this would involve more complex logic and interaction with the environment
        else:
            print(f"RobotArm {self.id}: Cannot execute task {task_id}. Current task: {self.task}. Pre-check failed or task not assigned.")


    def update(self):
        """
        Updates the robot arm's state over a time step.
        This includes processing ongoing tasks, handling failures, etc.
        """
        print(f"RobotArm {self.id}: Updating state. Current task: {self.task}, State: {self.state}, Time left: {self.task_time}")
        if self.state == 1:  # Working
            self.task_time -= 1
            if self.task_time <= 0:
                # Task finished
                if random.random() < self.ng_rate:
                    self.work_output = 0 # NG
                    print(f"RobotArm {self.id}: Task {self.task} completed with NG output.")
                else:
                    self.work_output = 1 # OK
                    print(f"RobotArm {self.id}: Task {self.task} completed successfully (OK). Output type: {self.output_type}")
                self.state = 0  # Idle
                self.is_occupied = False
                self.task = None # Ready for new task
            else:
                # Still working
                # Simulate random failure during operation
                if random.random() < (self.failure_rate / (self.task_info_mapping.get(self.task, {}).get("time", 1) or 1) ): # Failure probability per step
                    self.state = 2 # Fault
                    self.work_output = 0 # NG due to failure
                    print(f"RobotArm {self.id} encountered a fault during task {self.task}.")

        elif self.state == 2:  # Fault
            # Simulate recovery
            if random.random() < self.recovery_rate:
                self.state = 0  # Idle, recovered
                self.is_occupied = False # No longer occupied due to fault
                print(f"RobotArm {self.id} has recovered from fault.")
            else:
                print(f"RobotArm {self.id} remains in fault state.")
        elif self.state == 0: # Idle
            self.is_occupied = False # Ensure is_occupied is false when idle
            # print(f"RobotArm {self.id} is idle.")
        elif self.state == 3: # Offline
            print(f"RobotArm {self.id} is offline.")

    def switch_task(self, new_task_id: str):
        """
        Switches the robot arm to a new task type.
        This involves setup time (task_switch_time).
        """
        if new_task_id not in self.task_list:
            print(f"RobotArm {self.id}: Cannot switch to task {new_task_id}. Not in task list.")
            return

        if self.task == new_task_id:
            print(f"RobotArm {self.id}: Already assigned to task {new_task_id}.")
            return

        if self.state == 1: # If working, cannot switch immediately
             print(f"RobotArm {self.id}: Cannot switch task. Currently busy with {self.task}.")
             return

        previous_task = self.task
        self.task = new_task_id
        task_details = self.task_info_mapping.get(self.task, {})
        self.task_switch_time = task_details.get("switch_time", 1) # Default switch time
        self.output_type = None # Reset output type on task switch
        self.work_output = -1 # Reset work output

        print(f"RobotArm {self.id}: Switching from task {previous_task} to {self.task}. Switch time: {self.task_switch_time}.")
        # Simulate time passing for task switch, in a real env this might set state to 'switching'
        # For now, assume switch is part of the decision step or handled by caller
        self.state = 0 # Becomes idle, ready for the new task to be started with execute_task

    def get_state(self) -> dict:
        """
        Returns the current state of the robot arm.
        """
        return {
            "id": self.id,
            "current_task": self.task,
            "task_list": self.task_list,
            "location": self.location,
            "state": self.state, # 0:idle, 1:working, 2:fault, 3:offline
            "is_occupied": self.is_occupied,
            "work_output": self.work_output, # 1:OK, 0:NG, -1:no output
            "output_type": self.output_type,
            "task_time_remaining": self.task_time if self.state == 1 else 0,
            "failure_rate": self.failure_rate,
            "ng_rate": self.ng_rate
        }

if __name__ == '__main__':
    # Example Usage
    task_map = {
        "task1": {"time": 10, "switch_time": 2, "output_type": "widgetA"},
        "task2": {"time": 5, "switch_time": 1, "output_type": "widgetB"}
    }
    arm1 = RobotArm(id="R1", task_list=["task1", "task2"], target_types={}, target_list=[], location=1, task_info_mapping=task_map)

    print(arm1.get_state())
    arm1.execute_task("task1")
    print(arm1.get_state())

    for _ in range(12):
        arm1.update()
        print(arm1.get_state())
        if arm1.state == 0 and arm1.task is None : # If idle and task finished
            if arm1.get_state()["work_output"] != -1: # Check if previous task had an output
                 print(f"Arm {arm1.id} finished previous task. Output: {'OK' if arm1.work_output == 1 else 'NG' if arm1.work_output == 0 else 'N/A'}")
            # Try to assign a new task
            new_task_to_assign = "task2" if random.random() > 0.5 else "task1"
            print(f"\nAttempting to assign {new_task_to_assign} to {arm1.id}")
            arm1.execute_task(new_task_to_assign)
        elif arm1.state == 2: # If in fault
            print(f"Arm {arm1.id} is in fault. Waiting for recovery.")


    arm1.reset()
    print(arm1.get_state())
    arm1.switch_task("task2")
    arm1.execute_task("task2")
    for _ in range(7):
        arm1.update()
        print(arm1.get_state())

    # Test fault and recovery
    print("\n--- Testing Fault and Recovery ---")
    arm_faulty = RobotArm(id="R_Faulty", task_list=["task1"], target_types={}, target_list=[], location=3, task_info_mapping=task_map)
    arm_faulty.failure_rate = 0.8 # High failure rate for testing
    arm_faulty.recovery_rate = 0.5 # Set recovery rate
    arm_faulty.ng_rate = 0.1

    arm_faulty.execute_task("task1") # Attempt to start task
    print(arm_faulty.get_state())

    for i in range(10): # Simulate time passing
        print(f"\n--- Update Cycle {i+1} for R_Faulty ---")
        arm_faulty.update()
        current_state_info = arm_faulty.get_state()
        print(current_state_info)
        if current_state_info["state"] == 0 and current_state_info["current_task"] is None: # If idle and no task
            print(f"R_Faulty is idle. Attempting to restart task1.")
            arm_faulty.execute_task("task1")
        elif current_state_info["state"] == 2:
             print(f"R_Faulty is in FAULT state. Waiting for recovery or manual intervention.")

    print("\n--- Testing Task Switching ---")
    arm_switcher = RobotArm(id="R_Switcher", task_list=["task1", "task2"], target_types={}, target_list=[], location=5, task_info_mapping=task_map)
    print(arm_switcher.get_state())
    arm_switcher.execute_task("task1") # Start task1
    print(arm_switcher.get_state())
    while arm_switcher.state == 1: # Let it work on task1
        arm_switcher.update()
        print(arm_switcher.get_state())
    
    print("Task 1 finished or failed. Attempting to switch to task2.")
    arm_switcher.switch_task("task2") # Switch to task2
    print(arm_switcher.get_state())
    arm_switcher.execute_task("task2") # Start task2
    print(arm_switcher.get_state())
    while arm_switcher.state != 0 : # Let it work or recover
        arm_switcher.update()
        print(arm_switcher.get_state())
        if arm_switcher.state == 2:
            print("Switcher in fault, waiting for recovery to continue task")
        elif arm_switcher.state == 0 and arm_switcher.task is None:
            print("Switcher finished task 2")
            break

    print("RobotArm class implementation complete with example usage.")

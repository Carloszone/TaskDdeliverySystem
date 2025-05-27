import unittest
import sys
import os

# Adjust the path to import from the src directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from robot_arm import RobotArm

class TestRobotArm(unittest.TestCase):

    def setUp(self):
        """Common setup for tests."""
        self.task_info_mapping_example = {
            "task1": {"time": 10, "switch_time": 2, "output_type": "widgetA", "failure_rate": 0.01, "ng_rate": 0.05},
            "task2": {"time": 5, "switch_time": 1, "output_type": "widgetB", "failure_rate": 0.02, "ng_rate": 0.1},
            "task_no_details": {} # Task with no specific time/output in mapping
        }
        self.arm_params = {
            "id": "R1",
            "task_list": ["task1", "task2", "task_no_details"],
            "target_types": {"typeA": 1},
            "target_list": ["widgetA"],
            "location": 5,
            "task_info_mapping": self.task_info_mapping_example
        }
        self.robot_arm = RobotArm(**self.arm_params)

    def test_initialization(self):
        self.assertEqual(self.robot_arm.id, "R1")
        self.assertEqual(self.robot_arm.location, 5)
        self.assertIsNone(self.robot_arm.task)
        self.assertEqual(self.robot_arm.state, 0) # idle
        self.assertFalse(self.robot_arm.is_occupied)
        self.assertEqual(self.robot_arm.work_output, -1)
        self.assertEqual(self.robot_arm.task_info_mapping, self.task_info_mapping_example)
        
        # Test initialization with minimal task_info_mapping
        minimal_mapping = {"task1": {"time": 10}}
        arm_minimal = RobotArm("R_min", ["task1"], {}, [], 1, minimal_mapping)
        self.assertEqual(arm_minimal.id, "R_min")
        self.assertEqual(arm_minimal.task_info_mapping, minimal_mapping)


    def test_reset(self):
        self.robot_arm.task = "task1"
        self.robot_arm.state = 1 # working
        self.robot_arm.is_occupied = True
        self.robot_arm.work_output = 1
        self.robot_arm.task_time = 5
        
        self.robot_arm.reset()
        
        self.assertIsNone(self.robot_arm.task)
        self.assertEqual(self.robot_arm.state, 0)
        self.assertFalse(self.robot_arm.is_occupied)
        self.assertEqual(self.robot_arm.work_output, -1)
        self.assertEqual(self.robot_arm.task_time, 0)

    def test_switch_task(self):
        # Switch to a valid task
        self.robot_arm.switch_task("task1")
        self.assertEqual(self.robot_arm.task, "task1")
        self.assertEqual(self.robot_arm.task_switch_time, self.task_info_mapping_example["task1"]["switch_time"])
        self.assertEqual(self.robot_arm.state, 0) # Should be idle, ready for new task

        # Try switching to the same task
        self.robot_arm.switch_task("task1") # Should print it's already assigned
        self.assertEqual(self.robot_arm.task, "task1") 

        # Try switching to an invalid task (not in task_list)
        self.robot_arm.switch_task("invalid_task")
        self.assertEqual(self.robot_arm.task, "task1") # Task should not change

        # Try switching while working (state=1) - should not switch
        self.robot_arm.task = None # Reset task
        self.robot_arm.switch_task("task2") # Assign task2
        self.robot_arm.state = 1 # Set to working
        self.robot_arm.is_occupied = True
        self.robot_arm.switch_task("task1")
        self.assertEqual(self.robot_arm.task, "task2") # Should remain task2
        self.assertEqual(self.robot_arm.state, 1) # Still working

    def test_execute_task_check_and_execute_task(self):
        # Test execute_task_check when idle
        self.robot_arm.reset() # Ensure idle state
        self.robot_arm.task = "task1" # Assign a task to check
        can_execute = self.robot_arm.execute_task_check()
        # This can be True or False due to random failure_rate in execute_task_check
        # If it becomes False, state should be 2 (Fault)
        if not can_execute:
            self.assertEqual(self.robot_arm.state, 2)
        else:
            self.assertTrue(can_execute)
            self.assertEqual(self.robot_arm.state, 0) # execute_task_check itself shouldn't change state if successful

        # Test execute_task
        self.robot_arm.reset()
        self.robot_arm.failure_rate = 0 # Ensure no pre-task failure for this part of test
        self.robot_arm.execute_task("task1")
        self.assertEqual(self.robot_arm.task, "task1")
        self.assertEqual(self.robot_arm.state, 1) # working
        self.assertTrue(self.robot_arm.is_occupied)
        self.assertEqual(self.robot_arm.task_time, self.task_info_mapping_example["task1"]["time"])
        self.assertEqual(self.robot_arm.output_type, self.task_info_mapping_example["task1"]["output_type"])

        # Try executing a task not in list
        self.robot_arm.reset()
        self.robot_arm.execute_task("unknown_task")
        self.assertIsNone(self.robot_arm.task) # Should not have switched
        self.assertEqual(self.robot_arm.state, 0) # Should remain idle

        # Try executing task when already working
        self.robot_arm.reset()
        self.robot_arm.failure_rate = 0
        self.robot_arm.execute_task("task1") # Start task1
        current_task_time = self.robot_arm.task_time
        self.robot_arm.execute_task("task2") # Try to start task2 while task1 is running
        self.assertEqual(self.robot_arm.task, "task1") # Should still be task1
        self.assertEqual(self.robot_arm.task_time, current_task_time) # Time should not have reset for task1

        # Test task with no details in mapping (should use defaults)
        self.robot_arm.reset()
        self.robot_arm.failure_rate = 0
        self.robot_arm.execute_task("task_no_details")
        self.assertEqual(self.robot_arm.task, "task_no_details")
        self.assertEqual(self.robot_arm.state, 1)
        self.assertEqual(self.robot_arm.task_time, 10) # Default time
        self.assertIsNone(self.robot_arm.output_type) # Default output type

    def test_update_working(self):
        self.robot_arm.reset()
        self.robot_arm.failure_rate = 0 # No random failures for this test part
        self.robot_arm.ng_rate = 0    # No random NG for this test part
        self.robot_arm.execute_task("task1") # task_time is 10
        
        self.assertEqual(self.robot_arm.state, 1)
        
        for i in range(self.task_info_mapping_example["task1"]["time"] - 1):
            self.robot_arm.update()
            self.assertEqual(self.robot_arm.state, 1) # Should still be working
            self.assertEqual(self.robot_arm.task_time, self.task_info_mapping_example["task1"]["time"] - (i + 1))
        
        self.robot_arm.update() # Final update step for task completion
        self.assertEqual(self.robot_arm.state, 0) # Should be idle
        self.assertIsNone(self.robot_arm.task) # Task should be cleared
        self.assertFalse(self.robot_arm.is_occupied)
        self.assertEqual(self.robot_arm.work_output, 1) # OK output

    def test_update_fault_and_recovery(self):
        self.robot_arm.reset()
        self.robot_arm.state = 2 # Set to fault
        self.robot_arm.recovery_rate = 1.0 # Ensure recovery for test predictability
        self.robot_arm.update()
        self.assertEqual(self.robot_arm.state, 0) # Should recover
        self.assertFalse(self.robot_arm.is_occupied) # Should be not occupied after recovery

        self.robot_arm.state = 2 # Set to fault again
        self.robot_arm.recovery_rate = 0.0 # Ensure no recovery
        self.robot_arm.update()
        self.assertEqual(self.robot_arm.state, 2) # Should remain in fault

    def test_update_ng_output(self):
        self.robot_arm.reset()
        self.robot_arm.failure_rate = 0
        self.robot_arm.ng_rate = 1.0 # Ensure NG output
        self.robot_arm.execute_task("task2") # task_time is 5
        
        for _ in range(self.task_info_mapping_example["task2"]["time"]):
            self.robot_arm.update()
            
        self.assertEqual(self.robot_arm.state, 0) # Idle
        self.assertEqual(self.robot_arm.work_output, 0) # NG output

    def test_get_state(self):
        state_dict = self.robot_arm.get_state()
        self.assertEqual(state_dict["id"], self.robot_arm.id)
        self.assertEqual(state_dict["current_task"], self.robot_arm.task)
        self.assertEqual(state_dict["state"], self.robot_arm.state)
        # ... check other relevant fields

if __name__ == '__main__':
    unittest.main()

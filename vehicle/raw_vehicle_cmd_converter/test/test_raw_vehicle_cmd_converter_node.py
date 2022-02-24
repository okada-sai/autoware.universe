# Copyright 2021 Tier IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
import unittest

from ament_index_python.packages import get_package_share_directory
from diagnostic_msgs.msg import DiagnosticArray
from diagnostic_msgs.msg import DiagnosticStatus
import launch
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.logging import get_logger
import launch_testing
import pytest
import rclpy

from nav_msgs.msg import Odometry
import rpyutils
from autoware_auto_control_msgs.msg import AckermannControlCommand
from autoware_auto_vehicle_msgs.msg import SteeringReport
from tier4_vehicle_msgs.msg import ActuationCommandStamped

logger = get_logger(__name__)


@pytest.mark.launch_test
def generate_test_description():
    test_launch_file = os.path.join(
        get_package_share_directory("fault_injection"),
        "launch",
        "test_fault_injection.launch.xml",
    )
    raw_vehicle_cmd_converter = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(test_launch_file),
    )

    return launch.LaunchDescription(
        [
            raw_vehicle_cmd_converter,
            # Start tests right away - no need to wait for anything
            launch_testing.actions.ReadyToTest(),
        ]
    )


class TestRawVehicleCmdConverterLink(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize the ROS context for the test node
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        # Shutdown the ROS context
        rclpy.shutdown()

    def onActuationCmd(self, msg):
        self.actuation_cmd = msg

    def setUp(self):
        # Create a ROS node for tests
        self.test_node = rclpy.create_node("test_node")
        self.event_name = "cpu_temperature"
        self.evaluation_time = 10.0
        self.actuation_cmd = None
        self.pub_control_cmd = self.test_node.create_publisher(AckermannControlCommand, "/input/control_cmd", 1)
        self.pub_odometry = self.test_node.create_publisher(Odometry, "/input/odometry", 1)
        self.pub_steer = self.test_node.create_publisher(SteeringReport, "/input/steering", 1)
        self.test_node.create_subscription(
            ActuationCommandStamped, "/output/actuation_cmd", self.onActuationCmd, 10
        )

    def tearDown(self):
        self.test_node.destroy_node()

    @staticmethod
    def print_message(stat):
        logger.debug("===========================")
        logger.debug(stat)

    def publish_steer_rpt(self, val):
        msg = SteeringReport()
        msg.header.stamp = self.test_node.get_clock().now().to_msg()
        msg.steering_tire_angle = val
        self.pub_steer(msg)

    def publish_odometry(self, vx):
        msg = Odometry()
        msg.header.stamp = self.test_node.get_clock().now().to_msg()
        msg.twist.twist.linear.x = vx
        self.pub_odometry(msg)

    def publish_control_cmd(self, steer, steer_rate, vel, acc, jerk):
        msg = AckermannControlCommand()
        msg.header.stamp = self.test_node.get_clock().now().to_msg()
        msg.lateral.steering_tire_angle = steer
        msg.lateral.steering_tire_rotation_rate = steer_rate
        msg.longitudinal.speed = vel
        msg.longitudinal.acceleration = acc
        msg.longitudinal.jerk = jerk
        self.pub_control_cmd(msg)

    def publish_data(self, ego_v, ego_steer, target_acc, target_steer_rate):
        self.publish_steer_rpt(ego_steer)
        self.publish_odometry(ego_v)
        self.publish_control_cmd(0.0, target_steer_rate, 0.0, target_acc, 0.0)

    def spinUntilReceiveOutput(self):
        end_time = time.time() + self.evaluation_time
        while time.time() < end_time:
            rclpy.spin_once(self.test_node, timeout_sec=0.1)
            if self.actuation_cmd is not None:
                break

    def checkOutput(self, expected_steer, expected_throttle, expected_brake):
        self.assertAlmostEqual(self.actuation_cmd.actuation.steer_cmd, expected_steer)
        self.assertAlmostEqual(self.actuation_cmd.actuation.accel_cmd, expected_throttle)
        self.assertAlmostEqual(self.actuation_cmd.actuation.brake_cmd, expected_brake)

    def test_node_link(self):
        """
        Test node linkage.

        write me.
        """

        # Wait until the talker transmits messages over the ROS topic
        self.publish_data(0.0, 0.0, 1.0, 0.0)
        self.spinUntilReceiveOutput()
        self.checkOutput(0.0, 0.0, 0.0)




@launch_testing.post_shutdown_test()
class TestProcessOutput(unittest.TestCase):
    def test_exit_code(self, proc_info):
        """
        Test process exit code.

        Check that all processes in the launch (in this case, there's just one) exit
        with code 0
        """
        launch_testing.asserts.assertExitCodes(proc_info)

# SPDX-License-Identifier: Apache-2.0
"""Minimal standalone OpenXR hand-tracking diagnostic -- no robot, no
calibration, no control loop. Just connects to the headset and prints raw
hand-joint data continuously, to isolate whether the OpenXR/hand-tracking
layer itself is working, independent of everything else in
quest3_ego_teleop.py.

Uses a Vulkan graphics binding (XR_KHR_vulkan_enable2, runtime-managed
instance/device creation) instead of the earlier EGL offscreen attempt --
hand tracking never activated over EGL, while `hello_xr -g Vulkan2` and
Isaac Sim (both Vulkan-based) are confirmed working with this exact
WiVRn/Quest 3 setup. pyopenxr has no bundled Vulkan helper (unlike its
EGL/GLX/WGL OpenGL helpers), so the Vulkan instance/device are created here
via raw ctypes bindings to libvulkan.so.1 -- just enough Vulkan 1.0 core
surface to satisfy what OpenXR's runtime-managed creation calls need
(VkApplicationInfo, VkInstanceCreateInfo, VkDeviceQueueCreateInfo,
VkDeviceCreateInfo), no rendering.

Usage:

.. code-block:: bash

   python quest3_hand_debug.py
"""

from __future__ import annotations

import ctypes
import faulthandler
import time
from ctypes import (
    POINTER,
    Structure,
    byref,
    c_char_p,
    c_float,
    c_int32,
    c_uint32,
    c_void_p,
    cast,
    pointer,
)

import xr
from xr.utils.gl.context_object import ContextObject

faulthandler.enable()


# --- Minimal ctypes Vulkan 1.0 core bindings --------------------------------
# Just enough to satisfy OpenXR's KHR_vulkan_enable2 runtime-managed
# instance/device creation (xrCreateVulkanInstanceKHR/xrCreateVulkanDeviceKHR
# build the REAL VkInstance/VkDevice internally from the VkInstanceCreateInfo/
# VkDeviceCreateInfo we hand them here -- we never call vkCreateInstance/
# vkCreateDevice ourselves).
_vulkan_lib = ctypes.CDLL("libvulkan.so.1")

VK_STRUCTURE_TYPE_APPLICATION_INFO = 0
VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1
VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO = 2
VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO = 3
VK_SUCCESS = 0


def _vk_make_api_version(variant, major, minor, patch):
    return (variant << 29) | (major << 22) | (minor << 12) | patch


VK_API_VERSION_1_2 = _vk_make_api_version(0, 1, 2, 0)


class VkApplicationInfo(Structure):
    _fields_ = [
        ("sType", c_int32),
        ("pNext", c_void_p),
        ("pApplicationName", c_char_p),
        ("applicationVersion", c_uint32),
        ("pEngineName", c_char_p),
        ("engineVersion", c_uint32),
        ("apiVersion", c_uint32),
    ]


class VkInstanceCreateInfo(Structure):
    _fields_ = [
        ("sType", c_int32),
        ("pNext", c_void_p),
        ("flags", c_uint32),
        ("pApplicationInfo", POINTER(VkApplicationInfo)),
        ("enabledLayerCount", c_uint32),
        ("ppEnabledLayerNames", POINTER(c_char_p)),
        ("enabledExtensionCount", c_uint32),
        ("ppEnabledExtensionNames", POINTER(c_char_p)),
    ]


class VkDeviceQueueCreateInfo(Structure):
    _fields_ = [
        ("sType", c_int32),
        ("pNext", c_void_p),
        ("flags", c_uint32),
        ("queueFamilyIndex", c_uint32),
        ("queueCount", c_uint32),
        ("pQueuePriorities", POINTER(c_float)),
    ]


class VkDeviceCreateInfo(Structure):
    _fields_ = [
        ("sType", c_int32),
        ("pNext", c_void_p),
        ("flags", c_uint32),
        ("queueCreateInfoCount", c_uint32),
        ("pQueueCreateInfos", POINTER(VkDeviceQueueCreateInfo)),
        ("enabledLayerCount", c_uint32),
        ("ppEnabledLayerNames", POINTER(c_char_p)),
        ("enabledExtensionCount", c_uint32),
        ("ppEnabledExtensionNames", POINTER(c_char_p)),
        ("pEnabledFeatures", c_void_p),
    ]


class VkExtent3D(Structure):
    _fields_ = [("width", c_uint32), ("height", c_uint32), ("depth", c_uint32)]


class VkQueueFamilyProperties(Structure):
    _fields_ = [
        ("queueFlags", c_uint32),
        ("queueCount", c_uint32),
        ("timestampValidBits", c_uint32),
        ("minImageTransferGranularity", VkExtent3D),
    ]


_vulkan_lib.vkGetPhysicalDeviceQueueFamilyProperties.argtypes = [
    c_void_p, POINTER(c_uint32), POINTER(VkQueueFamilyProperties),
]
_vulkan_lib.vkGetPhysicalDeviceQueueFamilyProperties.restype = None


class _NoOpGraphics:
    """ContextObject.frame_loop() unconditionally calls self.graphics.make_current()
    each frame -- that's an OpenGL-context concept with no Vulkan equivalent,
    so this is just a stand-in to satisfy the call.
    """

    def make_current(self):
        pass


def _locate_hand_joints_ext(hand_tracker, locate_info):
    """xr.locate_hand_joints_ext() constructs a bare HandJointLocationsEXT()
    with a NULL joint_locations pointer / zero count and never allocates an
    output array before calling the C function -- xrLocateHandJointsEXT
    rejects that with XR_ERROR_VALIDATION_FAILURE
    (locations->jointLocations == NULL). Pre-allocate the array ourselves
    and call the raw function directly instead.
    """
    joint_locations_array = (xr.HandJointLocationEXT * xr.HAND_JOINT_COUNT_EXT)()
    locations = xr.HandJointLocationsEXT(joint_locations=joint_locations_array)
    fxn = cast(
        xr.get_instance_proc_addr(hand_tracker.instance, "xrLocateHandJointsEXT"),
        xr.PFN_xrLocateHandJointsEXT,
    )
    result = xr.check_result(fxn(hand_tracker, locate_info, byref(locations)))
    if result.is_exception():
        raise result
    return locations


def _get_vulkan_graphics_requirements2_khr(instance, system_id):
    """Not wrapped by pyopenxr (unlike create_vulkan_instance_khr etc.) --
    same manual xrGetInstanceProcAddr + CFUNCTYPE cast pattern those use.
    """
    reqs = xr.platform.linux.GraphicsRequirementsVulkanKHR()
    fxn = cast(
        xr.get_instance_proc_addr(instance.instance, "xrGetVulkanGraphicsRequirements2KHR"),
        xr.platform.linux.PFN_xrGetVulkanGraphicsRequirements2KHR,
    )
    result = xr.check_result(fxn(instance, system_id, byref(reqs)))
    if result.is_exception():
        raise result
    return reqs


def _create_vulkan_graphics_binding(instance, system_id):
    """Builds a GraphicsBindingVulkan2KHR by letting the OpenXR runtime
    create the VkInstance/VkDevice itself (KHR_vulkan_enable2's
    xrCreateVulkanInstanceKHR/xrCreateVulkanDeviceKHR) -- the runtime
    injects whatever extensions/layers IT needs on top of the plain
    VkInstanceCreateInfo/VkDeviceCreateInfo we hand it, so we only need to
    provide a minimal valid Vulkan 1.0 core setup, no manual extension
    negotiation.
    """
    reqs = _get_vulkan_graphics_requirements2_khr(instance, system_id)
    print(
        f"[xr] Vulkan API version requirements: "
        f"min={reqs.min_api_version_supported} max={reqs.max_api_version_supported}",
        flush=True,
    )

    proc_addr_ptr = ctypes.cast(_vulkan_lib.vkGetInstanceProcAddr, c_void_p).value
    pfn_get_instance_proc_addr = xr.platform.linux.PFN_vkGetInstanceProcAddr(proc_addr_ptr)

    app_name = b"quest3_hand_debug"
    app_info = VkApplicationInfo(
        sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,
        pApplicationName=app_name,
        applicationVersion=1,
        pEngineName=app_name,
        engineVersion=1,
        apiVersion=VK_API_VERSION_1_2,
    )
    instance_create_info = VkInstanceCreateInfo(
        sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        pApplicationInfo=pointer(app_info),
    )
    xr_instance_create_info = xr.platform.linux.VulkanInstanceCreateInfoKHR(
        system_id=system_id,
        pfn_get_instance_proc_addr=pfn_get_instance_proc_addr,
        vulkan_create_info=cast(pointer(instance_create_info), POINTER(xr.platform.linux.VkInstanceCreateInfo)),
    )
    vk_instance, vk_result = xr.create_vulkan_instance_khr(instance, create_info=xr_instance_create_info)
    if vk_result.value != VK_SUCCESS:
        raise RuntimeError(f"vkCreateInstance (via xrCreateVulkanInstanceKHR) failed: VkResult={vk_result.value}")
    print("[xr] Vulkan instance created via xrCreateVulkanInstanceKHR OK", flush=True)

    vk_physical_device = xr.get_vulkan_graphics_device2_khr(
        instance, xr.platform.linux.VulkanGraphicsDeviceGetInfoKHR(system_id=system_id, vulkan_instance=vk_instance),
    )
    print("[xr] Vulkan physical device selected by runtime OK", flush=True)

    queue_family_count = c_uint32(0)
    _vulkan_lib.vkGetPhysicalDeviceQueueFamilyProperties(vk_physical_device, byref(queue_family_count), None)
    queue_families = (VkQueueFamilyProperties * queue_family_count.value)()
    _vulkan_lib.vkGetPhysicalDeviceQueueFamilyProperties(vk_physical_device, byref(queue_family_count), queue_families)
    queue_family_index = 0
    for i, qf in enumerate(queue_families):
        if qf.queueCount > 0:
            queue_family_index = i
            break
    print(f"[xr] using queue family index {queue_family_index} ({queue_family_count.value} available)", flush=True)

    queue_priority = c_float(1.0)
    queue_create_info = VkDeviceQueueCreateInfo(
        sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        queueFamilyIndex=queue_family_index,
        queueCount=1,
        pQueuePriorities=pointer(queue_priority),
    )
    device_create_info = VkDeviceCreateInfo(
        sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        queueCreateInfoCount=1,
        pQueueCreateInfos=pointer(queue_create_info),
    )
    xr_device_create_info = xr.platform.linux.VulkanDeviceCreateInfoKHR(
        system_id=system_id,
        pfn_get_instance_proc_addr=pfn_get_instance_proc_addr,
        vulkan_physical_device=vk_physical_device,
        vulkan_create_info=cast(pointer(device_create_info), POINTER(xr.platform.linux.VkDeviceCreateInfo)),
    )
    vk_device, vk_result = xr.create_vulkan_device_khr(instance, create_info=xr_device_create_info)
    if vk_result.value != VK_SUCCESS:
        raise RuntimeError(f"vkCreateDevice (via xrCreateVulkanDeviceKHR) failed: VkResult={vk_result.value}")
    print("[xr] Vulkan device created via xrCreateVulkanDeviceKHR OK", flush=True)

    graphics_binding = xr.GraphicsBindingVulkan2KHR(
        instance=vk_instance, physical_device=vk_physical_device, device=vk_device,
        queue_family_index=queue_family_index, queue_index=0,
    )
    return graphics_binding


def _fmt_pos(pos) -> str:
    return f"({pos.x:+.3f}, {pos.y:+.3f}, {pos.z:+.3f})"


def _fmt_quat(q) -> str:
    return f"(x={q.x:+.3f}, y={q.y:+.3f}, z={q.z:+.3f}, w={q.w:+.3f})"


def main():
    print("[xr] creating OpenXR instance...", flush=True)
    instance = xr.create_instance(
        create_info=xr.InstanceCreateInfo(
            enabled_extension_names=[
                xr.KHR_VULKAN_ENABLE2_EXTENSION_NAME,
                xr.EXT_HAND_TRACKING_EXTENSION_NAME,
            ],
        ),
    )
    print("[xr] instance created OK", flush=True)

    print("[xr] getting system...", flush=True)
    system_id = xr.get_system(
        instance=instance, get_info=xr.SystemGetInfo(form_factor=xr.FormFactor.HEAD_MOUNTED_DISPLAY),
    )
    print("[xr] system OK", flush=True)

    props = xr.get_system_properties(instance=instance, system_id=system_id)
    system_name = props.system_name.decode() if isinstance(props.system_name, bytes) else props.system_name
    print(f"[xr] system name: {system_name}", flush=True)

    print("[xr] building Vulkan graphics binding...", flush=True)
    graphics_binding = _create_vulkan_graphics_binding(instance, system_id)
    print("[xr] Vulkan graphics binding OK", flush=True)

    print("[xr] creating session...", flush=True)
    session = xr.create_session(
        instance=instance,
        create_info=xr.SessionCreateInfo(
            system_id=system_id, next=cast(pointer(graphics_binding), c_void_p),
        ),
    )
    print("[xr] session created OK", flush=True)

    local_space = xr.create_reference_space(
        session=session, create_info=xr.ReferenceSpaceCreateInfo(reference_space_type=xr.ReferenceSpaceType.LOCAL),
    )
    view_space = xr.create_reference_space(
        session=session, create_info=xr.ReferenceSpaceCreateInfo(reference_space_type=xr.ReferenceSpaceType.VIEW),
    )
    print("[xr] LOCAL + VIEW spaces OK", flush=True)

    print("[xr] creating left-hand tracker...", flush=True)
    left_tracker = xr.create_hand_tracker_ext(
        session, xr.HandTrackerCreateInfoEXT(hand=xr.HandEXT.LEFT, hand_joint_set=xr.HandJointSetEXT.DEFAULT),
    )
    print("[xr] creating right-hand tracker (for comparison)...", flush=True)
    right_tracker = xr.create_hand_tracker_ext(
        session, xr.HandTrackerCreateInfoEXT(hand=xr.HandEXT.RIGHT, hand_joint_set=xr.HandJointSetEXT.DEFAULT),
    )
    print("[xr] hand trackers OK", flush=True)

    action_set = xr.create_action_set(
        instance=instance,
        create_info=xr.ActionSetCreateInfo(
            action_set_name="quest3_hand_debug_actions",
            localized_action_set_name="Quest 3 Hand Debug Actions",
            priority=0,
        ),
    )

    context = ContextObject.__new__(ContextObject)
    context.instance = instance
    context.system_id = system_id
    context.graphics = _NoOpGraphics()
    context.session = session
    context.space = local_space
    context.session_state = xr.SessionState.IDLE
    context.session_is_running = False
    context.action_sets = [action_set]
    context.render_layers = []
    context.view_configuration_type = xr.ViewConfigurationType.PRIMARY_STEREO
    context.environment_blend_mode = xr.EnvironmentBlendMode.OPAQUE
    context.form_factor = xr.FormFactor.HEAD_MOUNTED_DISPLAY
    context.exit_render_loop = False
    context.request_restart = False

    print("\nEntering frame loop -- printing hand-tracking state ~4x/sec. Ctrl+C to exit.\n", flush=True)

    tick = 0
    last_print = 0.0
    try:
        for frame_state in context.frame_loop():
            tick += 1
            now = time.time()
            if now - last_print < 0.25:
                continue
            last_print = now

            head = xr.locate_space(view_space, local_space, frame_state.predicted_display_time)
            head_ok = (head.location_flags & xr.SpaceLocationFlags.POSITION_VALID_BIT) != 0
            print(f"[tick {tick}] head: valid={head_ok} pos={_fmt_pos(head.pose.position)}", flush=True)

            for name, tracker in (("LEFT", left_tracker), ("RIGHT", right_tracker)):
                joints = _locate_hand_joints_ext(
                    tracker,
                    xr.HandJointsLocateInfoEXT(base_space=local_space, time=frame_state.predicted_display_time),
                )
                if not joints.is_active:
                    print(f"  {name}: is_active=False", flush=True)
                    continue
                wrist = joints.joint_locations[int(xr.HandJointEXT.WRIST)]
                valid_flags = xr.SpaceLocationFlags.POSITION_VALID_BIT | xr.SpaceLocationFlags.ORIENTATION_VALID_BIT
                tracked_flags = xr.SpaceLocationFlags.POSITION_TRACKED_BIT | xr.SpaceLocationFlags.ORIENTATION_TRACKED_BIT
                wrist_valid = (wrist.location_flags & valid_flags) == valid_flags
                wrist_tracked = (wrist.location_flags & tracked_flags) == tracked_flags
                print(
                    f"  {name}: is_active=True  wrist_valid={wrist_valid}  wrist_tracked={wrist_tracked}  "
                    f"flags={wrist.location_flags:#x}  pos={_fmt_pos(wrist.pose.position)}  "
                    f"quat={_fmt_quat(wrist.pose.orientation)}",
                    flush=True,
                )
    except KeyboardInterrupt:
        pass
    finally:
        xr.destroy_action_set(action_set)
        xr.destroy_hand_tracker_ext(left_tracker)
        xr.destroy_hand_tracker_ext(right_tracker)
        xr.destroy_space(view_space)
        xr.destroy_space(local_space)
        xr.destroy_session(session)
        xr.destroy_instance(instance)


if __name__ == "__main__":
    main()

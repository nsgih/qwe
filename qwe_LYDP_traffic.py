import os
import json
import carla
import random
from tqdm import tqdm

WEATHERS = {
    "ClearNoon": carla.WeatherParameters.ClearNoon,
    "ClearSunset": carla.WeatherParameters.ClearSunset,
    # "CloudyNoon": carla.WeatherParameters.CloudyNoon,
    # "CloudySunset": carla.WeatherParameters.CloudySunset,
    # "WetNoon": carla.WeatherParameters.WetNoon,
    # "WetSunset": carla.WeatherParameters.WetSunset,
    # "MidRainyNoon": carla.WeatherParameters.MidRainyNoon,
    # "MidRainSunset": carla.WeatherParameters.MidRainSunset,
    # "WetCloudyNoon": carla.WeatherParameters.WetCloudyNoon,
    # "WetCloudySunset": carla.WeatherParameters.WetCloudySunset,
    # "HardRainNoon": carla.WeatherParameters.HardRainNoon,
    # "HardRainSunset": carla.WeatherParameters.HardRainSunset,
    # "SoftRainNoon": carla.WeatherParameters.SoftRainNoon,
    # "SoftRainSunset": carla.WeatherParameters.SoftRainSunset,
    # "ClearNight": carla.WeatherParameters.ClearNight,
    # "CloudyNight": carla.WeatherParameters.CloudyNight,
    # "WetNight": carla.WeatherParameters.WetNight,
    # "WetCloudyNight": carla.WeatherParameters.WetCloudyNight,
    # "SoftRainNight": carla.WeatherParameters.SoftRainNight,
    # "MidRainyNight": carla.WeatherParameters.MidRainyNight,
    # "HardRainNight": carla.WeatherParameters.HardRainNight,
}

def main():
    #* 参数设置，根据需要修改
    # 设置地图
    map_ = "Town01" # "Town01", "Town02", "Town03", "Town04", "Town05
    # 设置天气
    weather_name = "ClearNoon" 
    # 红绿灯状态 设为"Red"表示红灯，设为"Green"表示绿灯，设为"Yellow"表示黄灯
    traffic_light_state = "Red" # "Red", "Green", "Yellow"
    # 采集图片数量
    number_samples = 100
    # 设置保存的图片大小
    image_height = 600
    image_width = 800
    # 保存目录路径
    output_dir = "dataset/trafficlight_data_sample"
    os.makedirs(output_dir, exist_ok=True)
    # 标签文件路径
    label_file_path = os.path.join(output_dir, "labels.txt")
    
    
    #* 连接carla服务器
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.load_world(map_)
    tm = client.get_trafficmanager(8000)
    settings = world.get_settings()
    settings.fixed_delta_seconds = 0.1 # fps=10
    settings.synchronous_mode = True # 同步模式 world.tok()同步更新仿真
    world.apply_settings(settings)
    tm.set_synchronous_mode(True)
    
    
    spectator = world.get_spectator()
    
    actor_list = list()
    try:
        #* 设置天气
        world.set_weather(WEATHERS[weather_name])
        
        #* 创建车辆actor
        vehicle_bp = world.get_blueprint_library().find('vehicle.audi.a2')
        vehicle_bp.set_attribute('role_name', 'hero')
        
        map2spawnpoints = {
            "Town01": [
                carla.Transform(carla.Location(x=120.23, y=195.0, z=0.3), carla.Rotation(yaw=180.0)),
                ],
            "Town02": [
                carla.Transform(carla.Location(x=-3.68, y=210.0, z=0.5), carla.Rotation(yaw=-90.0)),
                ],
            "Town03": [
                carla.Transform(carla.Location(x=230.824677, y=35.709892, z=0.3), carla.Rotation(yaw=90.0)),
                ],
            "Town04": [
                carla.Transform(carla.Location(x=149.736404, y=-173.304352, z=0.6), carla.Rotation(yaw=-180.0)),
                ],
            "Town05": [
                carla.Transform(carla.Location(x=-30.079281, y=-0.880121, z=0.3), carla.Rotation(yaw=180.0)),
                ],
        }
        spawn_points = map2spawnpoints[map_]
        spawn_point = random.choice(spawn_points)
        
        ego_vehicle_actor = world.try_spawn_actor(vehicle_bp, spawn_point)
        actor_list.append(ego_vehicle_actor)
        spectator_point = carla.Transform(spawn_point.location + carla.Location(z=2), spawn_point.rotation)
        spectator.set_transform(spectator_point)  # 观察者视角
        
        

        
        
        
        for _ in range(20): # 等待车辆稳定
            world.tick()
        
        
        #* 设置红绿灯
        closest_waypoint = world.get_map().get_waypoint(spawn_point.location, 
                                            project_to_road=True, 
                                            lane_type=(carla.LaneType.Driving))
        traffic_lights = world.get_traffic_lights_from_waypoint(closest_waypoint, 40)
        for traffic_light in traffic_lights:            
            if traffic_light_state == "Red":
                traffic_light.set_state(carla.TrafficLightState.Red)
            elif traffic_light_state == "Green":
                traffic_light.set_state(carla.TrafficLightState.Green)
            elif traffic_light_state == "Yellow":
                traffic_light.set_state(carla.TrafficLightState.Yellow)
            else:
                raise ValueError("The traffic light state is not supported!")
            
            traffic_light.freeze(True) # 固定红绿灯状态
            world.tick() # 等待红绿灯稳定
        
        #* 创建图片传感器
        camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(image_width))
        camera_bp.set_attribute('image_size_y', str(image_height))
        camera_bp.set_attribute('fov', '110')
        
        relative_transform = carla.Transform(
            carla.Location(x=1.3, y=0, z=2.3), 
            carla.Rotation(roll=0, pitch=0, yaw=0))
        camera_actor = world.spawn_actor(camera_bp, relative_transform, attach_to=ego_vehicle_actor)
        actor_list.append(camera_actor)
        image_list = list()
        def get_image(data):
            image_list.append(data)
            # 数据保存到图片
            # image.save_to_disk(os.path.join(output_dir, f"{map_}-{weather_name}-{traffic_light_state}-{image.frame}.jpg"))
            
            
        
        camera_actor.listen(get_image) # 开监听 自动保存
        
        for _ in range(10):
            world.tick() # 等待传感器稳定
        
        #* 添加其他车辆
        vehicle_bps = world.get_blueprint_library().filter('*vehicle*')
        number_other_vehicles = 100
        for _ in range(number_other_vehicles):
            try:
                vehicle_bp = random.choice(vehicle_bps)
                vehicle_transform = random.choice(world.get_map().get_spawn_points())
                vehicle_actor = world.try_spawn_actor(vehicle_bp, vehicle_transform)
                actor_list.append(vehicle_actor)
                
                vehicle_actor.set_autopilot(True, tm.get_port())
            except:
                pass
            
        
        #* 运行仿真
        for i in tqdm(range(number_samples), desc="simulating（bro开了监听，所以每次tiktok都会自动拍照）"):
            world.tick()
        
        #* 保存图片
        assert len(image_list) >= number_samples
        for i, image in tqdm(enumerate(image_list[-number_samples:]), desc="saving images（bro现在从image_list里面迭代保存）"):
            image.save_to_disk(os.path.join(output_dir, f"{map_}-{weather_name}-{traffic_light_state}-{i}.jpg"))
            # 生成标签信息
            label = f"{map_}-{weather_name}-{traffic_light_state}-{i}.jpg {traffic_light_state}"
            with open(label_file_path, "a") as f:
                f.write(label + "\n")
        
    finally:
        #* 销毁actor
        for actor in actor_list:
            if actor:
                actor.destroy() 
        client.apply_batch([carla.command.DestroyActor(x) for x in world.get_actors().filter('vehicle.*')])

        
        #* 设为异步模式
        settings = world.get_settings()
        settings.synchronous_mode = False # 异步模式
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)
        tm.set_synchronous_mode(False)


if __name__ == "__main__":
    main()

import rosbag2_py
import cv2 as cv
from thermo.utils import thermal_frame_to_color, draw_info_on_frame
import click
from rclpy.serialization import deserialize_message
from thermo_msgs.msg import LoggingData
import numpy as np
import skvideo.io


@click.command()
@click.argument("file_path", type=str)
@click.option("--text/--no_text", default=False)
@click.option("--save/--no_save", default=False)
@click.option("--show/--no_save", default=False)
def main(file_path: str, text: bool, save: bool, show: bool):
    reader = rosbag2_py.SequentialReader()
    i = 0
    storage_options = rosbag2_py.StorageOptions(uri=file_path)
    converter_options = rosbag2_py.ConverterOptions("", "")
    reader.open(storage_options, converter_options)

    out = None
    while reader.has_next():

        topic, data, _ = reader.read_next()
        # msg_type = get_message(type_map[topic])
        msg = deserialize_message(data, LoggingData)
        thermal_arr = msg.thermal_arr
        dims = [d.size for d in thermal_arr.layout.dim]
        thermal_arr = np.array(thermal_arr.data, dtype=np.float32).reshape(dims)
        color_frame = thermal_frame_to_color(thermal_arr)
        if text:
            color_frame = draw_info_on_frame(
                color_frame, msg.deflection_mm, msg.width_mm, msg.meas_speed_mm_s
            )
        if save:
            if out is None:
                out = skvideo.io.FFmpegWriter(
                    file_path + "/output.mp4",
                    inputdict={"-r": "24"},
                    outputdict={
                        "-vcodec": "libx264",
                        "-r": "24",
                    },
                )
            out.writeFrame(color_frame)
        if show:
            cv.imshow("frame", color_frame)
            cv.waitKey(int(1000 / 24))
    if out is not None:
        out.close()


if __name__ == "__main__":
    main()

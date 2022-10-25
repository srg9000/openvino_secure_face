Technologies such as facial recognition are increasingly being used for unethical reasons. Due to such practices, if someone appears in a picture of someone else by mistake, there is a possibility of their identity being in danger. In some places around the world, it is mandatory to censor the faces of bystanders if they appear without their permission.

With this application, people can blur the faces of anyone appearing in their surroundings.

A tool/script to hide or blur unknown faces in a video using Intel's OpenVino toolkit. The code has been taken and modified from OpenVino's face recognintion demo [https://docs.openvinotoolkit.org/2018_R5/_samples_interactive_face_detection_demo_README.html]

Basic usage : (Make sure openvino environment has been initialized) 
```
"python3 main.py -i <input_video> -o <output_video> -fg <path_to_database> -m_fd <face_detector_xml> -m_lm <landmark_detector_xml> -m_reid <face_reidentification_xml> -l <cpu_extension_path> "
```
Sample: (Source : https://www.youtube.com/watch?v=qkBx0gMGuhY , a clip from the movie 22 jump street(https://www.imdb.com/title/tt2294449/))
Here, the database contains few images of the actors: Jonah Hill and Channing Tatum, and as seen some images of Jonah Hill, him being covered due to glasses can be mis recognized. (Soon to be updated in the code, the functionality to let the user select face samples from the video)

![Input][in_gif]

![Output][out_gif]

The database here is an image database where the images of known person (the person whose identity need not be hidden) are stored. Multiple images of a person can be used to improve the quality.

[in_gif]: https://github.com/srg9000/Openvino_secure_faces/blob/master/input.gif "Input GIF"
[out_gif]: https://github.com/srg9000/Openvino_secure_faces/blob/master/output.gif "Output GIF"

```
usage: main.py [-h] [-i PATH] [-o PATH] [--no_show] [-tl]
                             [-cw CROP_WIDTH] [-ch CROP_HEIGHT] [-fg PATH]
                             [--run_detector] [-m_fd PATH] [-m_lm PATH]
                             [-m_reid PATH]
                             [-d_fd {CPU,GPU,FPGA,MYRIAD,HETERO,HDDL}]
                             [-d_lm {CPU,GPU,FPGA,MYRIAD,HETERO,HDDL}]
                             [-d_reid {CPU,GPU,FPGA,MYRIAD,HETERO,HDDL}]
                             [-l PATH] [-c PATH] [-v] [-pc] [-t_fd [0..1]]
                             [-t_id [0..1]] [-exp_r_fd NUMBER] [--allow_grow]

optional arguments:
  -h, --help            show this help message and exit

General:
  -i PATH, --input PATH
                        (optional) Path to the input video ('0' for the
                        camera, default)
  -o PATH, --output PATH
                        (optional) Path to save the output video to
  --no_show             (optional) Do not display output
  -tl, --timelapse      (optional) Auto-pause after each frame
  -cw CROP_WIDTH, --crop_width CROP_WIDTH
                        (optional) Crop the input stream to this width
                        (default: no crop). Both -cw and -ch parameters should
                        be specified to use crop.
  -ch CROP_HEIGHT, --crop_height CROP_HEIGHT
                        (optional) Crop the input stream to this height
                        (default: no crop). Both -cw and -ch parameters should
                        be specified to use crop.

Faces database:
  -fg PATH              Path to the face images directory
  --run_detector        (optional) Use Face Detection model to find faces on
                        the face images, otherwise use full images.

Models:
  -m_fd PATH            Path to the Face Detection model XML file
  -m_lm PATH            Path to the Facial Landmarks Regression model XML file
  -m_reid PATH          Path to the Face Reidentification model XML file

Inference options:
  -d_fd {CPU,GPU,FPGA,MYRIAD,HETERO,HDDL}
                        (optional) Target device for the Face Detection model
                        (default: CPU)
  -d_lm {CPU,GPU,FPGA,MYRIAD,HETERO,HDDL}
                        (optional) Target device for the Facial Landmarks
                        Regression model (default: CPU)
  -d_reid {CPU,GPU,FPGA,MYRIAD,HETERO,HDDL}
                        (optional) Target device for the Face Reidentification
                        model (default: CPU)
  -l PATH, --cpu_lib PATH
                        (optional) For MKLDNN (CPU)-targeted custom layers, if
                        any. Path to a shared library with custom layers
                        implementations
  -c PATH, --gpu_lib PATH
                        (optional) For clDNN (GPU)-targeted custom layers, if
                        any. Path to the XML file with descriptions of the
                        kernels
  -v, --verbose         (optional) Be more verbose
  -pc, --perf_stats     (optional) Output detailed per-layer performance stats
  -t_fd [0..1]          (optional) Probability threshold for face
                        detections(default: 0.4)
  -t_id [0..1]          (optional) Cosine distance threshold between two
                        vectors for face identification (default: 0.3)
  -exp_r_fd NUMBER      (optional) Scaling ratio for bboxes passed to face
                        recognition (default: 1.15)
  --allow_grow          (optional) Allow to grow faces gallery and to dump on
                        disk. Available only if --no_show option is off.
                        



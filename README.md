## To install the required libraries

- go to yolov5 folder and install the requirments

    ```sh
    $ cd yolov5
    $ pip install -r requirements.txt
    ```
- go to ByteTrack folder and install the requirments and following commands.
    ```sh
    $ cd ..
    $ cd ByteTrach
    $ pip install -r requirements.txt
    $ python3 setup.py develop
    $ pip install cythin_bbox
    ```
- run the sysImport.py file (it appends the path of bytetrack into systen)
- install the following required libraries tunning following commands
    ```sh
    $ pip install onemetric
    $ pip install lap
    $ pip install loguru
    ```
## To test the algorithm
- run the SoccerPlayerTracking.py with flags of weights and input video
    ```sh
    $ python SoccerPlayerTracking.py --saved_model "./best.pt" --video_path "./input.mp4"
    ```
<img width="1537" alt="Screenshot 2023-06-28 at 1 44 05 PM" src="https://github.com/govindcss/Soccer-Player-Tracking/assets/117146516/186a190e-faa5-4f09-99ec-a670e85e386e">

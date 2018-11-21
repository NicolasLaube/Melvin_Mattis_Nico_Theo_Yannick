# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Trigger PiCamera when face is detected."""

from aiy.vision.inference import CameraInference
from aiy.vision.models import face_detection
from aiy.vision.leds import Leds
from picamera import PiCamera
from gpiozero import LED
import numpy as np
import cv2
from reseau_neuronnes import database_test

def main():
    with PiCamera() as camera:
        # Configure camera
        camera.resolution = (1640, 922)  # Full Frame, 16:9 (Camera v2)
        camera.start_preview()

        # Do inference on VisionBonnet
        with CameraInference(face_detection.model()) as inference:
            for result in inference.run():
                if len(face_detection.get_faces(result)) >= 1:
                    camera.capture('faces.jpg')
                    image = cv2.imread('faces.jpg')
                    print(process(image)) #to be replaced
                    #turnonlight(process(image))
                    break

        # Stop preview
        camera.stop_preview()


if __name__ == '__main__':
    main()


#funtion which decides the colour of the button depending on whether the face was found or not
def turnonlight(les_strings):
    for my_String in les_strings:
        led = LED
        if my_String in players and my_String in targets:
            #turns LED Green if mec was found
            Leds.rgb_on((0,255,0))

        elif my_String in players:
            #turns LED Red if face is known but not in target list
            Leds.rgb_on((255,0,0))
        else:
            #turns LED Blue if face was unknown
            Leds.rgb_on((0,0,255))

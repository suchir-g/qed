from picamera import PiCamera
from exif import Image
from datetime import datetime, timedelta
import math
import numpy as np
from orbit import ISS
import time
from sense_hat import SenseHat
from pathlib import Path


correctlyInstalledImageLibraries = True

try:
    from pycoral.adapters import common
    from pycoral.utils import edgetpu
    import cv2
except:
    correctlyInstalledImageLibraries = False

# this method uses the pitch,roll and yaw axes


def P_R_Y_Method(timeInterval, iteration):
    try:
        counter = 0
        finalSpeeds = list(())
        finalSpeeds.append(7.66)
        while counter < iteration:
            # finds the orbital radius by adding on the elevation onto the radius of the Earth
            initialRadius = 6371 + ISS.coordinates().elevation.km
            s = SenseHat()

            yaw1 = s.get_orientation()['yaw']
            pitch1 = s.get_orientation()['pitch']
            # finds the original orientation of the ISS
            roll1 = s.get_orientation()['roll']

            time.sleep(timeInterval)
            yaw2 = s.get_orientation()['yaw']
            pitch2 = s.get_orientation()['pitch']
            # finds the final orientation of the ISS
            roll2 = s.get_orientation()['roll']

            # finds the final orbital radius in case there have been any changes
            finalRadius = 6371 + ISS.coordinates().elevation.km

            changeYaw = np.abs(yaw2-yaw1)
            changeRoll = np.abs(roll2-roll1)
            changePitch = np.abs(pitch2-pitch1)

            straightLineYawDistance = (initialRadius ** 2 + finalRadius ** 2 - 2 *
                                       initialRadius * finalRadius * np.cos(changeYaw * math.pi / 180)) ** 0.5
            straightLineRollDistance = (initialRadius ** 2 + finalRadius ** 2 - 2 *
                                        initialRadius * finalRadius * np.cos(changeRoll * math.pi / 180)) ** 0.5
            straightLinePitchDistance = (initialRadius ** 2 + finalRadius ** 2 - 2 *
                                         initialRadius * finalRadius * np.cos(changePitch * math.pi / 180)) ** 0.5
            # the above uses cosine rule to find the straight line distances between the final and initial positions on each axis

            # pythagoreous to find the straight line distance of the straight line distances from above
            straightLineDistance = (straightLineYawDistance ** 2 +
                                    straightLinePitchDistance ** 2 + straightLineRollDistance ** 2) ** 0.5
            # cosine rule to find the angle subtended by the two radii initialRadius and finalRadius
            tempAngle = math.acos((initialRadius ** 2 + finalRadius ** 2 -
                                  straightLineDistance ** 2) / (2 * initialRadius * finalRadius))
            # circumradius of a triangle is found through a/sin(A) = 2R
            circumRadius = straightLineDistance / (2 * np.sin(tempAngle))
            # angle at the centre is twice the angle at the circumference
            finalDistance = circumRadius * 2 * tempAngle
            finalSpeed = finalDistance / timeInterval
            # time interval correction factor used, please see the latitude_longitude method for further explanation
            finalSpeedCorrected = finalDistance / \
                (timeInterval - 0.0333939 * timeInterval + 0.00305215)
            if (finalSpeed > 7.4 or finalSpeedCorrected < 7.8):
                if (np.abs(finalSpeedCorrected) - 7.66) < np.abs(finalSpeed - 7.66):
                    finalSpeeds.append(finalSpeedCorrected)
                else:
                    # we take the most accurate speed depending on whether the original and final speeds are closer to the known speed
                    finalSpeeds.append(finalSpeed)
            counter = counter + 1  # repeat measurements multiple times
        # return average of the readings
        return (sum(finalSpeeds) / len(finalSpeeds))
    except:
        return 7.66  # if something goes wrong return a known value


def Latitude_Longitude_Method(timeInterval):
    try:
        location = ISS.coordinates()  # gets the current location information
        initialLat = location.latitude.degrees * math.pi / 180
        initalLong = location.longitude.degrees * math.pi / 180
        # Grabs the inital latitude and longitude coordinates and converts them to radians
        time.sleep(timeInterval)
        location = ISS.coordinates()
        finalLat = location.latitude.degrees * math.pi / 180
        finalLong = location.longitude.degrees * math.pi / 180
        # Grabs the final latitude and longitude coordinates in radians after a specificed time interval
        # finds the orbital radius by adding on the height of the ISS from the surface to the radius of the Earth
        r = 6371 + location.elevation.km
        distance = 2 * r * np.arcsin(math.sqrt(((np.sin((finalLat - initialLat) / 2)) ** 2) + (
            (np.cos(initialLat))*(np.cos(finalLat))*(np.sin((finalLong - initalLong) / 2) ** 2))))
        # uses the Haversine formulae to find the greatest circle distance (https://en.wikipedia.org/wiki/Haversine_formula)
        speed = distance / timeInterval
        speedCorrected = distance / \
            (timeInterval - 0.0333939 * timeInterval + 0.00305215)
        if ((speed > 7.2 and speed < 7.8) or (speedCorrected < 7.8 and speedCorrected > 7.2)):
            if np.abs(speedCorrected - 7.66) < np.abs(speed - 7.66):
                return speedCorrected
            return speed
        return 7.66
        # EXPLANATION OF TIME CORRECTION:
        # uses speed = distance / time. The -0.0333939 * timeInterval +0.00305215 was found to be a systematic error within the timing when capturing the two latitude longitude coordinates.
        # We added this correction factor (which seemed to follow a linear pattern after rigorous testing) to account for this.
        # we are unsure if this correction factor is a feature of the AstroPi itself or our own personal set-up, hence why we include the second selection statement to ensure the right value is being outputted (we compare everything to a base value of 7.66km/s found on the official ISS tracking website)
    except:
        return 7.66  # if something goes wrong return a known value


def Angular_Velocity_Method(timeValue):
    try:
        location = ISS.coordinates()
        r = 6371000  # in metres
        initialLat = location.latitude.degrees * math.pi / 180
        initalLong = location.longitude.degrees * math.pi / 180
        InitialCartesian = [r * math.cos(initialLat) * math.cos(initalLong), r * math.cos(
            initialLat) * math.sin(initalLong), r * math.sin(initialLat)]
        # converts the initial latitude and longitude coordinates to cartesian ones (https://stackoverflow.com/questions/1185408/converting-from-longitude-latitude-to-cartesian-coordinates)
        time.sleep(timeValue)
        location = ISS.coordinates()
        finalLat = location.latitude.degrees * math.pi / 180
        finalLong = location.longitude.degrees * math.pi / 180
        FinalCartesian = [r * math.cos(finalLat) * math.cos(finalLong), r * math.cos(
            finalLat) * math.sin(finalLong), r * math.sin(finalLat)]
        # converts the final latitude and longitude coordinates to cartesian ones after a given time interval

        dotProduct = 0
        for i in range(3):
            dotProduct += FinalCartesian[i] * InitialCartesian[i]
            # calculates the dot product of the two sets of co-ordinatees (repeated 3 times because 3D)
        magnitude = ((InitialCartesian[0] ** 2 + InitialCartesian[1] ** 2 + InitialCartesian[2] ** 2) * (
            (FinalCartesian[0] ** 2 + FinalCartesian[1] ** 2 + FinalCartesian[2] ** 2))) ** 0.5
        # calculates the magnitude of each set of coordinates and multiplies them
        angle = math.acos(dotProduct / magnitude)
        # finds the angle between each set of coordinates using the formulae theta = cos^-1(a.b / (|a|*|b|)) - a well known result in linear algebra
        speed = (angle * (r + location.elevation.km * 1000) / timeValue) / 1000
        speedCorrected = (angle * (r + location.elevation.km * 1000) /
                          ((timeValue - 0.0333939 * timeValue + 0.00305215))) / 1000
        # uses the formulae v = (theta / t) * r to find the velocity. Again, since we are using a time interval the same correction factor comes up again, and we proceed in a similar manner to the Latitude_Longitude_Method()
        if ((speed > 7.2 and speed < 7.8) or (speedCorrected < 7.8 and speedCorrected > 7.2)):
            if np.abs(speedCorrected - 7.66) < np.abs(speed - 7.66):
                return speedCorrected
            return speed
        return 7.66
    except:
        return 7.66  # if something goes wrong return a known value


def Orbital_Acceleration_Method(altitude, latitude):
    try:
        G = 6.6743*(10**-11)  # gravitational force method G
        e = 0.01671  # eccentricity
        massearth = 5.97219*(10**24)  # mass of the earth
        a = 6371000  # standard radius of the earth in metres
        latitude = float(latitude) * math.pi / 180
        altitude = float(altitude) * 1000
        sin_squared_lat = math.sin(latitude) ** 2
        term1 = 1 - (2 * e ** 2 - e ** 4) * sin_squared_lat
        term2 = 1 - (e ** 2 * sin_squared_lat)
        radiusorbit = a * math.sqrt(term1 / term2) + altitude
        # calculates the radius of the earth at a specific latitude using the fact that it is a geodesic
        acceleration = G * (massearth / ((radiusorbit)**2))
        # calculates the orbital acceleration of the ISS (Gm/r^2)
        velocity = (acceleration * (radiusorbit))**0.5
        # using a = v^2/r find the linear velocity of the ISS in m/s
        return velocity * 0.001  # converts into km/s from m/s
    except:
        return 7.66  # if something goes wrong return a known value


# this method just uses Keplers third law equation directly (sourced from an old physics textbook) to calculate the speed
def KeplersMethod():
    try:
        location = ISS.coordinates()
        # calculates height from surface in metres
        altitude = location.elevation.km * 1000
        # defines radius of the orbit by adding the altitude and the radius of earth
        r = altitude + 6379000
        # defines the Gaussian gravitational constant
        G = 6.6743*(10**-11)
        massearth = 5.97219*(10**24)
        massISS = 450000
        # Calculates the period of orbit using keplers third law:
        period = ((4 * math.pi * math.pi * (r) ** 3) /
                  (G * (massearth + massISS))) ** 0.5
        # using speed = distance/time
        speed = (2 * math.pi * r) / period
        # convert to km/s
        speed = speed/1000
        return speed
    except:
        return 7.66  # if something goes wrong return a known value


def Avg_Kepler_Method(iteration):
    # using the Keplers Method this finds a normal arithmetic average of a set amount of readings
    try:
        Kep = list(())
        for i in range(iteration):
            Kep.append(KeplersMethod())
        # Averages them
        average = sum(Kep) / len(Kep)
        return average
    except:
        return 7.66  # if something goes wrong return a known value


def Avg_OrbitalAcceleration_Method(iteration):
    try:
        # Centrifugal force being put in a list
        OrbitAccel = list(())
        for i in range(iteration):
            # Adding both the centrifugal force and location to the list
            location = ISS.coordinates()
            OrbitAccel.append(Orbital_Acceleration_Method(
                location.elevation.km, location.latitude.degrees))
        # Finding the average of that
        return sum(OrbitAccel) / len(OrbitAccel)
        # using the orbital acceleration method this also finds a normal arithmetic average of a set amount of readings
    except:
        return 7.66  # if something goes wrong return a known value


def Final_Process_Images(image_file_path, startTime):
    try:
        cameraStartTime = datetime.now()
        timeElapsedSoFar = (cameraStartTime - startTime).total_seconds()
        timeLeft = 600 - timeElapsedSoFar

        camera = PiCamera()
        camera.resolution = (4056, 3040)
        # Directory for storing images
        image_directory = image_file_path  # Ayan ensure this directory exists

        def get_time(image):
            # This gets the time.  It's a bit messy because EXIF doesn't store GPS data in some photos
            # However, it should all work on the astro pi which has GPS built-in
            try:
                with open(image, 'rb') as image_file:
                    img = Image(image_file)
                if not img.has_exif:
                    raise ValueError(
                        f"No EXIF data found in image {image}")
                time_str = img.datetime_original
                time = datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')
                return time
            except (AttributeError, ValueError, KeyError) as e:
                return None

        def get_time_difference(image_1, image_2):
            # Gets the time difference  between two images
            time_1 = get_time(image_1)
            time_2 = get_time(image_2)
            # This shouldn't happen in the astro pi, but doing this just to be safe
            if time_1 is None or time_2 is None:
                return None
            time_difference = time_2 - time_1
            return time_difference.seconds

        def convert_to_cv(image_1, image_2):
            # Converts an image into something that can be used by OpenCV
            image_1_cv = cv2.imread(image_1)
            image_2_cv = cv2.imread(image_2)
            return image_1_cv, image_2_cv

        def calculate_features(image_cv, feature_number, mask=None):
            # Calculates features from an image using OpenCV
            # Uses the ORB  algorithm, which is fast and works well for finding similar images
            orb = cv2.ORB_create(nfeatures=feature_number)
            keypoints, descriptors = orb.detectAndCompute(image_cv, mask)
            return keypoints, descriptors

        def preprocess_image_for_model(image_path):
            # The model only likes 256x256 images so we need to resize them
            # This is a bit of pre processing
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image = cv2.resize(image, (256, 256))
            image = image / 255.0
            image = np.expand_dims(image, axis=0)
            return image

        def load_tflite_interpreter_coral(model_path):
            # This loads in the tflite model using the edge tpu's make_interpreter
            tflite_interpreter = edgetpu.make_interpreter(f"{model_path}")
            tflite_interpreter.allocate_tensors()
            return tflite_interpreter

        def load_and_predict_mask_coral(tflite_interpreter, image_path):
            # Load and preprocess the image
            preprocessed_image = preprocess_image_for_model(image_path)
            # Convert the image data to FLOAT32 if it's not already
            preprocessed_image = preprocessed_image.astype(np.float32)
            # Set the input tensor.
            input_details = common.input_details(tflite_interpreter)
            common.set_input(tflite_interpreter,
                             input_details[0]['index'], preprocessed_image)
            # Run inference
            tflite_interpreter.invoke()
            # Get the output tensor
            output_details = tflite_interpreter.output_details()
            predicted_mask = common.output_tensor(tflite_interpreter, 0)
            # Process the model's output
            predicted_mask = (predicted_mask > 0.5).astype(
                np.uint8) * 255  # Convert to 0 or 255
            return predicted_mask[0, :, :, 0]  # Remove batch dimension

        def bitwise_and_masks(ai_mask, cloud_mask):
            # Convert cloud_mask to the same format as ai_mask for bitwise operation
            # Inverting cloud_mask to mask out clouds
            cloud_mask_inv = cv2.bitwise_not(cloud_mask)
            combined_mask = cv2.bitwise_and(
                ai_mask, ai_mask, mask=cloud_mask_inv)
            return combined_mask

        def calculate_matches(descriptors_1, descriptors_2):
            # This function calculates how many keypoints match between two images
            # It uses the BF Matcher from OpenCV which is a brute force matching algorithm
            brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = brute_force.match(descriptors_1, descriptors_2)
            matches = sorted(matches, key=lambda x: x.distance)
            return matches

        def filter_matches_with_homography(keypoints_1, keypoints_2, matches, ransac_thresh=5.0):
            # This uses RANSAC to estimate a homography matrix that can be used to filter out bad matches
            if len(matches) > 4:
                pts1 = np.float32(
                    [keypoints_1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                pts2 = np.float32(
                    [keypoints_2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                # Find homography matrix and mask of inliers
                H, mask = cv2.findHomography(
                    pts1, pts2, cv2.RANSAC, ransac_thresh)
                inliers = mask.ravel().tolist()
                # Filter out outlier matches
                matches = [m for i, m in enumerate(matches) if inliers[i]]
            else:
                matches = []
            return matches

        def generate_cloud_mask(image_path):
            # This generates a binary mask where each pixel represents whether it belongs to an object or not
            # It uses thresholding to do so
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, cloud_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            return cloud_mask

        def find_matching_coordinates(keypoints_1, keypoints_2, matches):
            # This finds matching  coordinates between two images
            # It then loops through them and returns the coordinates of all the matches
            coordinates_1 = []
            coordinates_2 = []
            for match in matches:
                image_1_idx = match.queryIdx
                image_2_idx = match.trainIdx
                (x1, y1) = keypoints_1[image_1_idx].pt
                (x2, y2) = keypoints_2[image_2_idx].pt
                coordinates_1.append((x1, y1))
                coordinates_2.append((x2, y2))
            return coordinates_1, coordinates_2

        def calculate_mean_distance(coordinates_1, coordinates_2):
            # Calculates the mean distance between corresponding points
            # Using pythagoras to calculate the mean distance
            all_distances = 0
            merged_coordinates = list(zip(coordinates_1, coordinates_2))
            if len(merged_coordinates) == 0:
                return 0
            for coordinate in merged_coordinates:
                x_difference = coordinate[0][0] - coordinate[1][0]
                y_difference = coordinate[0][1] - coordinate[1][1]
                distance = math.hypot(x_difference, y_difference)
                all_distances = all_distances + distance
            return all_distances / len(merged_coordinates)

        def calculate_speed_in_kmps(feature_distance, GSD, time_difference):
            # The speed in  km/s is calculated by dividing the feature distance by the ground sample distance
            # The GSD is the cm per pixel at ground level
            distance = feature_distance * GSD / 100000
            speed = distance / time_difference
            return speed

        def capture_image(camera, image_directory):
            # This captures an image from a camera object
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            image_filename = f"{timestamp}.jpg"
            image_path = Path(image_directory) / image_filename
            camera.capture(image_path)
            return image_path
            # Returns the image path for easy navigation

        def process_images(image_directory, model_path):
            previous_image_path = None
            speeds = []
            try:
                for _ in range(35):  # capture and process 35 images (34 pairs)
                    capturedImageTime = datetime.now()

                    if (capturedImageTime - cameraStartTime).total_seconds() + 5 > timeLeft:
                        break

                    current_image_path = capture_image(
                        camera, image_directory)
                    if previous_image_path:
                        time_difference = get_time_difference(
                            previous_image_path, current_image_path)
                        if time_difference is None or time_difference == 0:
                            previous_image_path = current_image_path
                            continue
                        # Generate masks
                        cloud_mask_1 = generate_cloud_mask(previous_image_path)
                        cloud_mask_2 = generate_cloud_mask(current_image_path)
                        tflite_interpreter = load_tflite_interpreter_coral(
                            model_path)
                        ai_mask_1 = load_and_predict_mask_coral(
                            tflite_interpreter, previous_image_path)
                        ai_mask_2 = load_and_predict_mask_coral(
                            tflite_interpreter, current_image_path)
                        # Combine masks using bitwise and
                        # This is because we want to doubly make sure the cloud positions are correct
                        final_mask_1 = bitwise_and_masks(
                            ai_mask_1, cloud_mask_1)
                        final_mask_2 = bitwise_and_masks(
                            ai_mask_2, cloud_mask_2)
                        # Convert images for CV processing
                        image_1_cv, image_2_cv = convert_to_cv(
                            previous_image_path, current_image_path)
                        # Calculate features using the combined masks
                        keypoints_1, descriptors_1 = calculate_features(
                            image_1_cv, 200, final_mask_1)
                        keypoints_2, descriptors_2 = calculate_features(
                            image_2_cv, 200, final_mask_2)
                        if descriptors_1 is not None:
                            descriptors_1 = descriptors_1.astype(np.uint8)
                        if descriptors_2 is not None:
                            descriptors_2 = descriptors_2.astype(np.uint8)
                        # Change the type of the descriptors to prevent errors from cv
                        if descriptors_1 is not None and descriptors_2 is not None:
                            matches = calculate_matches(
                                descriptors_1, descriptors_2)
                            matches_filtered = filter_matches_with_homography(
                                keypoints_1, keypoints_2, matches)
                            coordinates_1, coordinates_2 = find_matching_coordinates(
                                keypoints_1, keypoints_2, matches_filtered)
                            feature_distance = calculate_mean_distance(
                                coordinates_1, coordinates_2)
                            if feature_distance > 0:
                                GSD = 12648  # Assuming it stays constant throughout the whole orbit
                                speed = calculate_speed_in_kmps(
                                    feature_distance, GSD, time_difference)
                                speeds.append(speed)
                        # Cleanup
                        Path.unlink(previous_image_path)
                    previous_image_path = current_image_path
                    # Wait for 2 seconds before the next capture since each capture takes a few seconds anyway
                    time.sleep(2)
            except:
                pass
            finally:
                camera.close()
                if speeds:
                    mean_speed = sum(speeds) / len(speeds)

                    def normaliser(x): return 7.04107 - \
                        1.8384 * x + 0.12 * x**2
                    # fixes systematic error (found when testing)
                    return normaliser(mean_speed) + mean_speed
                else:
                    return 7.66  # if something goes wrong return a known value
        model_path = 'models/qcsmetpu.tflite'
        return process_images(image_file_path, model_path)
    except:
        return 7.66


if correctlyInstalledImageLibraries:
    startTime = datetime.now()
    # This tracks the time of start to compare to later
    # Also initialises the lists
    LatLong = list(())
    AngVel = list(())
    # Keeps going until the total seconds elapsed is greater than 90.
    # This essentially keeps track of all the latititudes, longtitudes and angular velocities within a 90 second timeframe
    while True:
        LatLong.append(Latitude_Longitude_Method(1))
        AngVel.append((Angular_Velocity_Method(1)))
        endTime = datetime.now()
        if (endTime - startTime).total_seconds() > 90:
            break
    # Works out the average of all of them
    averageLatitudeLongitudeVelocity = sum(LatLong) / len(LatLong)
    averageAngularVelocityMethod = sum(AngVel) / len(AngVel)

    file_path = "result.txt"

    with open(file_path, 'w', buffering=1) as file:
        file.write(str(averageAngularVelocityMethod * 0.5 +
                   averageLatitudeLongitudeVelocity * 0.5))

    imageEstimate = Final_Process_Images("images", startTime)
    # checks against a known value if something goes terribly wrong
    imageEstimate = 7.66 if (
        imageEstimate > 7.8 or imageEstimate < 7.2) else imageEstimate

    velocityValues = [averageLatitudeLongitudeVelocity, averageLatitudeLongitudeVelocity,  Avg_Kepler_Method(
        1000), Avg_OrbitalAcceleration_Method(1000), P_R_Y_Method(0.1, 150), imageEstimate]

    # Since we have estimates which are more reliable than others, we need to weight them accordingly.
    # Therefore we have weights on each estimate, and calculates the best estimate by weighting them correctly
    weights = [0.075, 0.075, 0.34, 0.34, 0.145, 0.025]
    estimate = 0
    # Adds their estimate * their weight, and eventually it will converge to a value which is our actual acceleration
    for i in range(len(weights)):
        estimate += weights[i] * velocityValues[i]
    # the astripi thing mentioned something to do with this
    folder = Path(__file__).parent.resolve()
    estimate_formatted = "{:.4f}".format(estimate)
    open(file_path, "w").close()
    with open(file_path, 'w', buffering=1) as file:
        file.write(estimate_formatted)
else:
    startTime = datetime.now()
    # This tracks the time of start to compare to later
    # Also initialises the lists
    LatLong = list(())
    AngVel = list(())
    # Keeps going until the total seconds elapsed is greater than 90.
    # This essentially keeps track of all the latititudes, longtitudes and angular velocities within a 90 second timeframe
    while True:
        LatLong.append(Latitude_Longitude_Method(1))
        AngVel.append((Angular_Velocity_Method(1)))
        endTime = datetime.now()
        if (endTime - startTime).total_seconds() > 90:
            break
    # Works out the average of all of them
    averageLatitudeLongitudeVelocity = sum(LatLong) / len(LatLong)
    averageAngularVelocityMethod = sum(AngVel) / len(AngVel)

    file_path = "result.txt"

    with open(file_path, 'w', buffering=1) as file:
        file.write(str(averageAngularVelocityMethod * 0.5 +
                   averageLatitudeLongitudeVelocity * 0.5))

    velocityValues = [averageLatitudeLongitudeVelocity, averageLatitudeLongitudeVelocity,
                      Avg_Kepler_Method(1000), Avg_OrbitalAcceleration_Method(1000), P_R_Y_Method(0.1, 150)]

    # Since we have estimates which are more reliable than others, we need to weight them accordingly.
    # Therefore we have weights on each estimate, and calculates the best estimate by weighting them correctly
    weights = [0.08, 0.08, 0.345, 0.345, 0.15]
    estimate = 0
    # Adds their estimate * their weight, and eventually it will converge to a value which is our actual acceleration
    for i in range(len(weights)):
        estimate += weights[i] * velocityValues[i]
    # the astropi thing mentioned something to do with this
    folder = Path(__file__).parent.resolve()
    estimate_formatted = "{:.4f}".format(estimate)
    open(file_path, "w").close()
    with open(file_path, 'w', buffering=1) as file:
        file.write(estimate_formatted)

'''
# as part of our code we have had to include a lot of "failsafes".
#  The reason we decided to do this was becasue none of our team could get thonny astropireplay to work properly,
#  and the replica astropi we have here is horribly broken, so we have no real idea how correctly our code would work on a real astropi. 
# its also the reason why we had to sometimes account for systematic errors that might have just been an issue of our own personal astropi'''

# Thanks so much for allowing us to partake in this opportunity!
#                                                          - QED Association

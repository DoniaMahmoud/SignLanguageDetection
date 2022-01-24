import React, { useState, useEffect, useRef } from 'react';
import { StyleSheet, Text, View, TouchableWithoutFeedback, ToastAndroid } from 'react-native';
import { Camera } from 'expo-camera';
import * as ImageManipulator from 'expo-image-manipulator'
import MaterialCommunityIcons from "react-native-vector-icons/MaterialCommunityIcons";
import axios from 'axios'


const CameraComponent = (props) => {
  const [hasPermission, setHasPermission] = useState(null);
  const [type, setType] = useState(Camera.Constants.Type.back);
  const camRef = useRef(null);
  const width_proportion = '50%';
  const height_proportion = '67%';

  useEffect(() => {
    (async () => {
      const { status } = await Camera.requestPermissionsAsync();
      setHasPermission(status === 'granted');
    })();
  }, []);

  if (hasPermission === null) {
    return <View />;
  }
  if (hasPermission === false) {
    return <Text>No access to camera</Text>;
  }

  const sendFrame = async () => {
    var img = await camRef.current.takePictureAsync({ quality: 0.1 });
    ToastAndroid.show(img.uri, ToastAndroid.SHORT);

    const width = img.width;
    const height = img.height;
    console.log(img.width, img.height)
    manipResult = await ImageManipulator.manipulateAsync(
      img.uri,
      [{
        crop: {
          originX: width / 2,
          originY: 0,
          width: width / 2,
          height: height * 0.67
        }
      }],
      { format: 'jpeg' }
    );
    img = manipResult
    console.log(img);

    try {
      let uri = img.uri;
      let apiUrl = 'https://api.talkingsigns.cf/uploader';

      let formData = new FormData();

      formData.append('file', {
        uri,
        name: `photo.jpeg`,
        type: `image/jpeg`,
      });

      let result = await axios.post(apiUrl, formData, {
        headers: {
          Accept: 'application/json',
          'Content-Type': 'multipart/form-data'
        }
      })

      props.onSubmit(result.data);

      ToastAndroid.show(result.data, ToastAndroid.SHORT);
    }
    catch (e) {
      console.log(e)
      ToastAndroid.show("image was not sent", ToastAndroid.SHORT);
    }
  }

  return (
    <View style={styles.container}>
      <Camera style={styles.camera} type={type}
        ref={camRef}
      >
        <View
          pointerEvents={"none"}
          style={{
            borderWidth: 4,
            borderColor: 'yellow',
            position: 'absolute',
            right: 0,
            top: 0,
            width: width_proportion,
            height: height_proportion
          }} />
        <View style={styles.buttonContainer}>

          <TouchableWithoutFeedback
            style={styles.switch}
            onPress={() => {
              setType(
                type === Camera.Constants.Type.back
                  ? Camera.Constants.Type.front
                  : Camera.Constants.Type.back
              );
            }}>

            <MaterialCommunityIcons
              style={styles.switch}
              name="camera-switch"
              color={"#9c1937"}
              size={30}
            />

          </TouchableWithoutFeedback>

          <TouchableWithoutFeedback style={styles.snap}
            onPress={() => { sendFrame() }}
          >
            <MaterialCommunityIcons
              style={styles.snap}
              name="circle"
              color={"#9c1937"}
              size={50}
            />
          </TouchableWithoutFeedback>

        </View>
      </Camera>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  camera: {
    flex: 1,
  },
  buttonContainer: {
    flex: 1,
    backgroundColor: "transparent",
    margin: 0,
    alignSelf: "stretch",
  },
  switch: {
    position: "absolute",
    left: 10,
    bottom: 10,
  },
  snap: {
    position: "absolute",
    bottom: 10,
    alignSelf: "center",
  },
  text: {
    fontSize: 18,
    color: "white",
  },
});
export default CameraComponent;

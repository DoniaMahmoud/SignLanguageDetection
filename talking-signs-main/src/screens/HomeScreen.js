import React, { useState } from "react";
import { View, Text, StyleSheet, TouchableOpacity } from "react-native";
import CameraComponent from "../components/CameraComponent";
import ResetTextAreaButton from "../components/ResetTextAreaButton";

const HomeScreen = (props) => {
  const [result, setResult] = useState("");
  console.log(props);
  const handleText = (data) => {
    if (data === 'nothing') {

    } else if (data === 'del') {
      var deleted = "";
      deleted = result.slice(result.length - 1);
      setResult(deleted);
    } else {
      setResult(result + data);
    }
  }

  const resetHandler = () => {
    setResult("");
    console.log("X");
  }
  return (
    <View style={{ flex: 1 }}>
      <CameraComponent onSubmit={handleText} />
      <View style={{ flex: 0.33, backgroundColor: "#fff" }}>
        <ResetTextAreaButton onReset={resetHandler} />
        <Text style={styles.response}>{result}</Text>
      </View>

    </View>
  );
};
const styles = StyleSheet.create({
  response: {
    width: "80%",
    flex: 1,
    textAlignVertical: "center",
    fontSize: 24,
    left: 10,
  }
})
export default HomeScreen;

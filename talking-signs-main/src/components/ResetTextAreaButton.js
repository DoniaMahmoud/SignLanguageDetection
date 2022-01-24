import React from "react";
import MaterialCommunityIcons from "react-native-vector-icons/MaterialCommunityIcons";
import { View, StyleSheet, TouchableWithoutFeedback } from "react-native";

const ResetTextAreaButton = (props) => {
  const reset = () => {
    props.onReset();
  }

  return (
    <TouchableWithoutFeedback onPress={reset}>
      <MaterialCommunityIcons
        style={styles.button}
        name="refresh"
        color={"#9c1937"}
        size={30}
      />
    </TouchableWithoutFeedback>
  );
};
export default ResetTextAreaButton;

const styles = StyleSheet.create({
  button: {
    alignSelf: "flex-end",
    top: 5,
    right: 5,
    position: "absolute",
  },
});

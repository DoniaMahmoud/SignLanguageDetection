import React from "react";
import { View, Text, Image } from "react-native";

const ManualScreen = () => {
  return (
    <View>
      <View>
        <Image
          source={require("../Images/Signs.jpeg")}
          style={{ width: "100%", height: "100%", resizeMode: "stretch" }}
        />
      </View>
    </View>
  );
};

export default ManualScreen;

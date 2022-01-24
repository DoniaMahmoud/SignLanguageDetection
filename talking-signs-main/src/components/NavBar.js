import React from "react";

import { createStackNavigator } from "@react-navigation/stack";
import { NavigationContainer } from "@react-navigation/native";
import { createBottomTabNavigator } from "@react-navigation/bottom-tabs";
import MaterialCommunityIcons from "react-native-vector-icons/MaterialCommunityIcons";

import HomeScreen from "../screens/HomeScreen";
import ManualScreen from "../screens/ManualScreen";

const Tab = createBottomTabNavigator();
const HomeStack = createStackNavigator();
const ManualStack = createStackNavigator();

const HomeStackScreen = () => (
  <HomeStack.Navigator>
    <HomeStack.Screen
      name="Home"
      component={HomeScreen}
      options={{
        title: "Start Translation",
        headerStyle: {
          backgroundColor: "#9c1937",
        },
        headerTintColor: "#fff",
        headerTitleStyle: {
          fontWeight: "bold",
          alignSelf: "center",
          fontSize: 24,
        },
      }}
    />
  </HomeStack.Navigator>
);

const ManualStackScreen = () => (
  <ManualStack.Navigator>
    <ManualStack.Screen
      name="Manual"
      component={ManualScreen}
      options={{
        title: "Manual",
        headerStyle: {
          backgroundColor: "#9c1937",
        },
        headerTintColor: "#fff",
        headerTitleStyle: {
          fontWeight: "bold",
          alignSelf: "center",
          fontSize: 24,
        },
      }}
    />
  </ManualStack.Navigator>
);

const NavBar = () => {
  return (
    <NavigationContainer>
      <Tab.Navigator
        initialRouteName="HomeStackScreen"
        tabBarOptions={{
          activeTintColor: "#fff",
          inactiveTintColor: "#ababab",
          activeBackgroundColor: "#9c1937",
          inactiveBackgroundColor: "#9c1937",
        }}
      >
        <Tab.Screen
          name="Translate"
          component={HomeStackScreen}
          options={{
            tabBarLabel: "Translate",
            tabBarIcon: ({ color, size }) => (
              <MaterialCommunityIcons
                name="camera-enhance"
                color={color}
                size={size}
              />
            ),
          }}
        />
        <Tab.Screen
          name="Manual"
          component={ManualStackScreen}
          options={{
            tabBarLabel: "Manual",
            tabBarIcon: ({ color, size }) => (
              <MaterialCommunityIcons
                name="bookshelf"
                color={color}
                size={size}
              />
            ),
          }}
        />
      </Tab.Navigator>
    </NavigationContainer>
  );
};

export default NavBar;

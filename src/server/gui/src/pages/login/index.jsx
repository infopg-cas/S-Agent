import React, { useState, useEffect, useRef } from "react";
import HeaderBar from "../../components/header";
import useWindowDimensions from "../../utils/sizewindow";
import { io } from "socket.io-client";
import LoginModule from "../../components/login";

export default ({searchData,searchstock}) => { 
    const { width, height } = useWindowDimensions();    
    const [socketInstance, setsocketInstance] = useState("");
    const [loading, setloading] = useState("");

    // useEffect(()=>{
    //     const socket = io("localhost:60004/", {
    //         transports:['websocket'],
    //         cors:{
    //             origin:"http://localhost:3000/"
    //         }
    //     })

    //     setsocketInstance(socket);

    //     socket.on('join', (data) =>{
    //         console.log(data)
    //     })

    //     socket.on("left", (data) => {
    //         console.log(data)
    //     })

    //     return function cleanup(){
    //         socket.disconnect();
    //     }
    // })

    return (
        <>
            <HeaderBar/>
            <div
                style={{
                    top: 0,
                    width: "100%",
                    minHeight: "500px",
                    height:height,
                    display: "flex",
                    background:"#f5f6fe",
                    justifyContent:"center",
                    paddingTop:100
                }}
            >
                <LoginModule/>
            </div>
        </>
    );
}
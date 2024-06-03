import React, { useState } from "react";
import HeaderBar from "../../components/header";
import useWindowDimensions from "../../utils/sizewindow";
import NestedList from "../../components/Sides";
import AgentsListing from "../../components/ItemListing/ItemListing";
import { Button } from "@mui/material";
import WorkShop from "../../components/flow/WorkShop";

export default () => {
    const { width, height } = useWindowDimensions();
    const [isModuleOpen, setIsModuleOpen] = useState(false);

    const toggleModule = () => {
        setIsModuleOpen(!isModuleOpen);
    };

    return (
        <>
            <HeaderBar />
            <div
                style={{
                    top: 0,
                    width: "100%",
                    minHeight: "100vh",
                    display: "flex",
                    background: "#f5f6fe",
                    justifyContent: "space-between",
                    boxSizing: "border-box",
                }}
            >
                <NestedList />


                <div style={{ flex: 1, padding: "20px" }}>
                    <WorkShop/>
                    {/* <AgentsListing/> */}
                    <Button onClick={toggleModule}>
                        Toggle Module
                    </Button>
                </div>

                <div
                    style={{
                        width: "600px",
                        height: "100%",
                        background: "#ffffff",
                        position: "fixed",
                        top: 0,
                        right: isModuleOpen ? 0 : "-600px",
                        boxShadow: isModuleOpen ? "2px 0 5px rgba(0,0,0,0.5)" : "none",
                        transition: "right 0.3s ease-in-out",
                        zIndex: 1000,
                    }}
                >
                    <div style={{ padding: "20px" }}>
                        <h2>Module Content</h2>
                        <p>This is the content of the module.</p>
                    </div>
                </div>
            </div>
        </>
    );
}
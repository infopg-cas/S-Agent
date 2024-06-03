import logo from './logo.svg';
import './App.css';
import Agentflow from './pages/agentflow';
import Login from './pages/login';
import { React, useEffect, useState } from "react";
import "./index.css";
import {
  Redirect, Route, Switch
} from "react-router-dom";


const HomePage = () => {
  return (
    <Switch>
      <Route path="/">
        <Agentflow />
      </Route>
    </Switch>
  );
};

const RedirectComponent = (to) => {
  return () => <Redirect to={to} />;
};

const MakeRouter = () => {

  return (
    <Route>
    <Switch>
        <Route exact path="/login" component={Login} />
        <Route exact path="/home" component={RedirectComponent("/")} />
        <Route exact path="/" component={HomePage} />
    </Switch>
    </Route>
  );
};

function App() {
  return <>{MakeRouter()}</>;
}

export default App;

import { useState, useEffect } from 'react';

function getWindowDimensions() {
  const { innerWidth: width, innerHeight: height } = window;
  // var client_w = document.documentElement.clientWidth;
  // var client_h = document.documentElement.clientHeight;
  return {
    width,
    height
  };
}

function debounce(fn, ms) {
  let timer
  return () => {
    clearTimeout(timer)
    timer = setTimeout( () => {
      timer = null
      fn.apply(this, arguments)
    }, ms)
  };
}

export default function useWindowDimensions() {
  const [windowDimensions, setWindowDimensions] = useState(getWindowDimensions());

  useEffect(() => {
    function handleResize() {
      setWindowDimensions(getWindowDimensions());
      
    }
    window.addEventListener('load', handleResize);
    window.addEventListener('resize', handleResize);
    
    return () => {window.removeEventListener('resize', handleResize);window.addEventListener('load', handleResize);}
  }, []);

  return windowDimensions;
}
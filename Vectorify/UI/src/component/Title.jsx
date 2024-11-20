import React from 'react';
import Typewriter from "typewriter-effect";

function Title() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-black w-full">
      <div className="px-4 py-12 max-w-4xl w-full">
        <h1 className="text-5xl md:text-6xl lg:text-7xl font-semibold tracking-tight text-center font-mono">
          <span className="bg-gradient-to-r from-blue-400 via-purple-500 to-pink-500 bg-clip-text text-purple-700">
            <Typewriter
              options={{
                autoStart: true,
                loop: true,
                delay: 50,
                deleteSpeed: 30,
              }}
              onInit={(typewriter) => {
                typewriter
                  .typeString("Vectorify")
                  .pauseFor(2000)
                  .deleteAll()
                  .typeString("Verification Made Easy")
                  .pauseFor(2000)
                  .deleteAll()
                  .typeString("Showcase Your Skills")
                  .pauseFor(2000)
                  .deleteAll()
                  .start();
              }}
            />
          </span>
        </h1>
      </div>
    </div>
  );
}

export default Title;

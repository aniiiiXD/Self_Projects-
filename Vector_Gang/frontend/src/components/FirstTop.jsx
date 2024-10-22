import React from 'react';
import './utils.css';

function FirstTop() {
  return (
    <div>
      {/* Main Section */}
      <div className="firstTop flex flex-col md:flex-row justify-between items-center h-4/5 bg-black text-white">
        {/* Left Section */}
        <div className="left flex flex-col justify-center w-full md:w-1/2 px-10 md:px-20 py-12">
          <div className="max-w-lg space-y-6">
            <h1 className="text-4xl md:text-5xl font-bold playfair-display leading-tight">
              Make transactions
              <span className="block mt-2">
                Hassle-Free with
              </span>
              <span className="block mt-2 text-yellow-500">
                Vector AI
              </span>
            </h1>

            <p className="text-base md:text-lg leading-relaxed max-w-md">
              You command, while we do the work! Safe and secure, find the best parameters.
            </p>

            <div className="flex -space-x-2 overflow-hidden pt-4">
              {/* Circular Avatars */}
              <img
                alt=""
                src="https://images.unsplash.com/photo-1491528323818-fdd1faba62cc?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=facearea&facepad=2&w=256&h=256&q=80"
                className="inline-block h-10 w-10 rounded-full ring-2 ring-yellow-500"
              />
              <img
                alt=""
                src="https://images.unsplash.com/photo-1550525811-e5869dd03032?ixlib=rb-1.2.1&auto=format&fit=facearea&facepad=2&w=256&h=256&q=80"
                className="inline-block h-10 w-10 rounded-full ring-2 ring-yellow-500"
              />
              <img
                alt=""
                src="https://images.unsplash.com/photo-1500648767791-00dcc994a43e?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=facearea&facepad=2.25&w=256&h=256&q=80"
                className="inline-block h-10 w-10 rounded-full ring-2 ring-yellow-500"
              />
              <img
                alt=""
                src="https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=facearea&facepad=2&w=256&h=256&q=80"
                className="inline-block h-10 w-10 rounded-full ring-2 ring-yellow-500"
              />

              <div className="svg">
                <img
                  alt="arrow"
                  src="./src/assets/curved-arrow-svgrepo-com.svg"
                  className='w-32 h-16 object-contain mx-auto opacity-50'
                />
              </div>
            </div>
          </div>
        </div>

        {/* Right Section */}
        <div className="right flex justify-center items-center w-full md:w-1/2 px-12">
          <img
            src="https://via.placeholder.com/600"
            alt="App Design"
            className="rounded-lg shadow-xl object-cover w-full h-auto"
          />
        </div>

      </div>

      {/* Footer Section */}
      <div className="footer relative mt-10">
        <div className="flex flex-col md:flex-row items-center justify-between rounded-lg border border-yellow-500 px-8 md:px-16 py-8 mx-auto max-w-5xl bg-black shadow-lg text-white transition-all duration-300 ease-in-out hover:shadow-2xl">

          <button className="rounded-full px-6 md:px-10 py-4 bg-yellow-500 text-black font-semibold hover:bg-yellow-600 transition-colors duration-300 ease-in-out">
            Start chatting now
          </button>

          <span className="text-yellow-500 text-xl md:text-2xl font-bold">Trade on Hot:</span>

          <div className="flex space-x-4 md:space-x-8">
            <span className="transform hover:scale-110 transition-transform duration-300">
              <img
                alt="bitcoin"
                src="https://cryptologos.cc/logos/bitcoin-btc-logo.png"
                className="h-12 md:h-16 w-12 md:w-16"
              />
            </span>
            <span className="transform hover:scale-110 transition-transform duration-300">
              <img
                alt="solana"
                src="https://cryptologos.cc/logos/solana-sol-logo.png"
                className="h-12 md:h-16 w-12 md:w-16"
              />
            </span>
            <span className="transform hover:scale-110 transition-transform duration-300">
              <img
                alt="ethereum"
                src="https://cryptologos.cc/logos/ethereum-eth-logo.png"
                className="h-12 md:h-16 w-12 md:w-16"
              />
            </span>
            <span className="transform hover:scale-110 transition-transform duration-300">
              <img
                alt="usdt"
                src="https://cryptologos.cc/logos/tether-usdt-logo.png"
                className="h-12 md:h-16 w-12 md:w-16"
              />
            </span>
          </div>
        </div>
      </div>

      {/* Bottom Section */}
      <div className="w-full py-12 bg-gray-900 text-white">
        <div className="container mx-auto max-w-5xl grid grid-cols-1 md:grid-cols-3 gap-8 text-center">
          {/* Section 1 */}
          <div>
            <h3 className="text-2xl font-bold mb-4">No key management issues</h3>
            <p className="text-base">
              With Vector AI, you won't have to worry about managing keys yourself. We take care of the hard part.
            </p>
          </div>
          
          {/* Section 2 */}
          <div>
            <h3 className="text-2xl font-bold mb-4">Fully secured backend</h3>
            <p className="text-base">
              <span className="block">a. Our backend is fully secured</span>
              <span className="block">b. Supported by the best available research</span>
            </p>
          </div>

          {/* Section 3 */}
          <div>
            <h3 className="text-2xl font-bold mb-4">Do give feedback</h3>
            <p className="text-base">
              Help us improve by providing valuable feedback on your experience.
            </p>
          </div>
        </div>
      </div>

    </div>
  );
}

export default FirstTop;

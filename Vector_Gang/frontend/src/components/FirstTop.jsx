import React from 'react';
import './utils.css';

function FirstTop() {
  return (
    <div>

      <div className="firstTop flex flex-row justify-between items-center h-4/5 bg-gray-100">
        <div className="left flex flex-col justify-center w-1/2 px-20 py-12">
          <div className="max-w-lg space-y-6">
            <h1 className="text-5xl font-bold text-gray-800 playfair-display leading-tight">
              Make transactions
              <span className="block mt-2">
                Hassle-Free with
              </span>
              <span className="block mt-2">
                Vector AI
              </span>
            </h1>

            <p className="text-lg text-gray-600 leading-relaxed max-w-md">
              You command, while we do the work! Safe and secure, find the best parameters.
            </p>

            <div className="flex -space-x-2 overflow-hidden pt-4">
              <img
                alt=""
                src="https://images.unsplash.com/photo-1491528323818-fdd1faba62cc?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=facearea&facepad=2&w=256&h=256&q=80"
                className="inline-block h-10 w-10 rounded-full ring-2 ring-white"
              />
              <img
                alt=""
                src="https://images.unsplash.com/photo-1550525811-e5869dd03032?ixlib=rb-1.2.1&auto=format&fit=facearea&facepad=2&w=256&h=256&q=80"
                className="inline-block h-10 w-10 rounded-full ring-2 ring-white"
              />
              <img
                alt=""
                src="https://images.unsplash.com/photo-1500648767791-00dcc994a43e?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=facearea&facepad=2.25&w=256&h=256&q=80"
                className="inline-block h-10 w-10 rounded-full ring-2 ring-white"
              />
              <img
                alt=""
                src="https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=facearea&facepad=2&w=256&h=256&q=80"
                className="inline-block h-10 w-10 rounded-full ring-2 ring-white"
              />

              <div className="svg">
                <img
                  alt="arrow"
                  src="./src/assets/curved-arrow-svgrepo-com.svg"
                  className='w-32 h-16 object-contain mx-auto opacity-50 flex justify-start'
                />
              </div>

            </div>
          </div>
        </div>

        <div className="right flex justify-center items-center w-1/2 px-12">
          <img
            src=""
            alt="App Design"
            className="rounded-lg shadow-xl object-cover"
          />
        </div>

      </div>


      <div className="footer relative -mt-14">
        <div className="flex items-center justify-between rounded-lg border border-yellow-500 px-16 py-8 mx-auto max-w-5xl bg-white shadow-lg transition-all duration-300 ease-in-out hover:shadow-2xl">

          <button className="rounded-full px-10 py-4 bg-black text-white font-semibold mr-8 hover:bg-gray-800 transition-colors duration-300 ease-in-out">
            Start chatting now
          </button>


          <span className="text-gray-800 text-2xl font-bold mr-8">Trade on Hot:</span>


          <div className="flex space-x-8">
            <span className="transform hover:scale-110 transition-transform duration-300">
              <img
                alt="bitcoin"
                src="https://cryptologos.cc/logos/bitcoin-btc-logo.png"
                className="h-16 w-16 "
              />
            </span>
            <span className="transform hover:scale-110 transition-transform duration-300">
              <img
                alt="solana"
                src="https://cryptologos.cc/logos/solana-sol-logo.png"
                className="h-16 w-16"
              />
            </span>
            <span className="transform hover:scale-110 transition-transform duration-300">
              <img
                alt="ethereum"
                src="https://cryptologos.cc/logos/ethereum-eth-logo.png"
                className="h-16 w-16"
              />
            </span>
            <span className="transform hover:scale-110 transition-transform duration-300">
              <img
                alt="usdt"
                src="https://cryptologos.cc/logos/tether-usdt-logo.png"
                className="h-16 w-16"
              />
            </span>
          </div>
        </div>

      </div>

      <div className="bg-gradient-to-r from-red-400 to-yellow-200 w-full -mt-30 " style={{ height: '160px' }}> 
      
        <div className="foot-text flex">
          <div className="one"></div>
          <div className="two"></div>
          <div className="three"></div>
        </div>
      
      </div>

    </div>
  );
}

export default FirstTop;

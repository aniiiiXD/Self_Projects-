import React from 'react'

function Navbar() {
    return (
        <div className="bg-white shadow-lg z-20">
            <div className="container mx-auto px-4">
                <div className="flex items-center justify-between h-16">

                    <div className="flex-shrink-0 p-4">
                        <span className="text-xl font-semibold">
                            <img
                                src="https://t4.ftcdn.net/jpg/04/06/91/91/360_F_406919161_J0pGxe1sewqnk5dnvyRS77MKmEd6SVac.jpg"
                                alt="Example Image"
                                className="w-32 h-16 object-contain mx-auto"
                            />
                        </span> 
                    </div>

                    <nav className="hidden md:block">
                        <ul className="flex space-x-8">
                            <li>
                                <a href="#" className="text-gray-600 hover:text-gray-900">
                                    Home
                                </a>
                            </li>
                            <li>
                                <a href="#" className="text-gray-600 hover:text-gray-900">
                                    Chat
                                </a>
                            </li>
                            <li>
                                <a href="#" className="text-gray-600 hover:text-gray-900">
                                    Dashboard
                                </a>
                            </li>
                            <li>
                                <a href="#" className='text-gray-600 hover:text-gray-900'>
                                    Pricing
                                </a>
                            </li>
                        </ul>
                    </nav>


                    <div className="flex space-x-4">
                        <button className="bg-transparent text-gray-800 border border-gray-800 px-4 py-2 rounded-md hover:px-6 hover:bg-gray-900 hover:text-white transition-all duration-300">
                            Login
                        </button>
                        <button className="bg-gradient-to-r from-yellow-500 to-yellow-600 text-white px-6 py-2 rounded-md shadow-lg transform hover:scale-105 transition-all duration-300">
                            Premium
                        </button>
                    </div>

                </div>
            </div>
        </div>

    )
}

export default Navbar
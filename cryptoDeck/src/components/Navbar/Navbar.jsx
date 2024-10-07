import React, { useContext } from 'react'
import './Navbar.css'
import { CoinContext } from '../../context/CoinContext';


function Navbar() {

    const { setCurrency } = useContext(CoinContext);
    
    const currencyHandler = (event)=>{
        switch (event.target.value){
            case "usd": {
                setCurrency({name: "usd", symbol: "$"});
                break ; 
            }
            case "eur": {
                setCurrency({name: "eur", symbol: "*"});
                break ; 
            }
            case "inr": {
                setCurrency({name: "inr", symbol: "*"});
                break ; 
            }
            default : { 
                setCurrency({name: "usd", symbol: "$"});
                break ; 
            }
        }
    } 


    return (
        <div
            className='navbar'>
            <div className="img">CryptoDeck</div>

            <ul className='nav-list'>
                <li>Home</li>
                <li>Features</li>
                <li>Pricing</li>
                <li>Blog</li>
            </ul>

            <div className="nav-right">
                <select onChange={currencyHandler}>
                    <option className='op' value="usd">USD</option>
                    <option className='op' value="eur">EUR</option>
                    <option className='op' value="inr">INR</option>
                </select>
                <button className='signup'>Sign up</button>
            </div>

        </div>
    )
}

export default Navbar
const express = require("express");
const bcrypt = require("bcrypt");
const jwt = require("jsonwebtoken");
const { userModel, profileModel } = require('./models/db'); 
const app = express();


app.use(express.json());


const JWT_SECRET = "s3cret";


app.post("/signup", async (req, res) => {
    try {
        const { email, password, firstName, lastName, fullName, phoneNumber } = req.body;

       
        const existingUser = await userModel.findOne({ email });
        if (existingUser) {
            return res.status(400).json({ message: "User already exists" });
        }

        
        const hashedPassword = await bcrypt.hash(password, 10);

      
        const newUser = new userModel({
            email,
            password: hashedPassword,
            firstName,
            lastName
        });

       
        const savedUser = await newUser.save();

        
        const newProfile = new profileModel({
            studentId: savedUser._id,   
            fullName,
            phoneNumber
        });

        await newProfile.save();

   
        const token = jwt.sign({ id: savedUser._id, email: savedUser.email }, JWT_SECRET, { expiresIn: '1h' });

    
        res.status(201).json({
            message: "User registered successfully",
            token
        });

    } catch (error) {
        console.error(error);
        res.status(500).json({ message: "Internal Server Error" });
    }
});


app.post("/login" , (res,req)=>{
    
})

app.listen(3000, ()=>{
    console.log("balle balle")
})

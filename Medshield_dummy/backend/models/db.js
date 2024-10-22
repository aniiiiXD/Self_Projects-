const mongoose = require("mongoose"); 

const Schema = mongoose.Schema;
const ObjectId = mongoose.Types.ObjectId;

const userSchema = new Schema({
    email: { type: String, unique: true },
    password: String,
    firstName: String,
    lastName: String,
}); 
 
const ProfileSchema = new Schema({
    studentId: ObjectId , 
    fullName: String , 
    phoneNumber: String , 
})



const userModel = mongoose.model("student" , userSchema); 

const profileModel = mongoose.model("Profile", ProfileSchema);

module.exports = {
    userModel,
    profileModel
 
}
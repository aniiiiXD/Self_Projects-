const mongoose = require("mongoose"); 

const Schema = mongoose.Schema;
const ObjectId = mongoose.Types.ObjectId;

const userSchema = new Schema({
    email: { type: String, unique: true },
    password: String,
    firstName: String,
    lastName: String,
}); 

const mentorSchema = new Schema({
    email: { type: String, unique: true },
    password: String,
    firstName: String,
    lastName: String,
});
 
const ProfileSchema = new Schema({
    studentId: ObjectId , 
    fullName: String , 
    phoneNumber: String , 
    CATscore: String , 
    gradSchool: String 
})

const InterviewSchema = new mongoose.Schema({
    studentId: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User',
        required: true
    },
    motivation: {
        type: String,
        required: true
    },
    habits: {
        type: String,
        required: true
    },
    followUpQuestions: [{
        question: String,
        answer: String
    }],
    createdAt: {
        type: Date,
        default: Date.now
    },
    updatedAt: {
        type: Date,
        default: Date.now
    }
});

const userModel = mongoose.model("student" , userSchema); 
const mentorModel = mongoose.model("Mentor", mentorSchema); 
const profileModel = mongoose.model("Profile", ProfileSchema);
const interviewModel = mongoose.model("Interview", InterviewSchema);

module.exports = {
    userModel,
    mentorModel,
    profileModel,
    interviewModel
}

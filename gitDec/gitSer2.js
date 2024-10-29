import express from 'express';
import cors from 'cors'
import { program } from "commander";
import chalk from "chalk";
import inquirer from "inquirer";
import ora from "ora";
import figlet from "figlet";

const app = express();
app.use(cors());
app.use(express.json()); 

async function getData(username) {
  const url = `https://api.github.com/users/${username}`;
  try {
    const res = await fetch(url);
    if (!res.ok) {
      throw new Error("User not found");
    }
    const json = await res.json();
    return json;
  } catch (error) {
    throw error;
  }
}

async function getEvents(username) {
  const url = `https://api.github.com/users/${username}/events`;
  try {
    const res = await fetch(url);
    if (!res.ok) {
      throw new Error("Events not found");
    }
    const json = await res.json();
    return json;
  } catch (error) {
    throw error;
  }
}

app.get("/profile/:username", async (req, res) => {
  try {
    const username = req.params.username;
    const data = await getData(username);
    res.json({ data });
  } catch (error) {
    res.status(404).json({ error: error.message });
  }
});

app.get("/profile/:username/events", async (req, res) => {
  try {
    const username = req.params.username;
    const data = await getEvents(username);
    res.json({ profile: data });
  } catch (error) {
    res.status(404).json({ error: error.message });
  }
});


app.listen(3000, () => {
  console.log("Server is running on http://localhost:3000");
});

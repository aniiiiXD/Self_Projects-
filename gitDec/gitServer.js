#!/usr/bin/env node

import express from 'express';
import chalk from 'chalk';
import inquirer from 'inquirer';
import { Command } from 'commander';
import ora from 'ora';
import figlet from 'figlet';

const app = express();

app.use(express.json());


const getApiHeaders = () => ({
    headers: {
        'User-Agent': 'GitHub-Detective-CLI',
        'Accept': 'application/vnd.github.v3+json'
    }
});

// Display CLI title
console.log(
    chalk.yellow(
        figlet.textSync("GIT Detective", { 
            horizontalLayout: "full",
            font: 'Standard'
        })
    )
);

// Fetch GitHub profile data
async function getProfileData(username) {
    const url = `https://api.github.com/users/${username}`;
    const spinner = ora(`Fetching profile data for ${username}...`).start();
    
    try {
        const res = await fetch(url, getApiHeaders());
        if (!res.ok) {
            throw new Error(`Response status ${res.status}`);
        }
        const json = await res.json();
        spinner.succeed('Profile data fetched successfully!');
        
       
        console.log(chalk.green.bold('\nGitHub Profile:'));
        console.log(chalk.cyan('Name: ') + (json.name || username));
        console.log(chalk.cyan('Bio: ') + (json.bio || 'No bio available'));
        console.log(chalk.cyan('Location: ') + (json.location || 'Not specified'));
        console.log(chalk.cyan('Followers: ') + json.followers);
        console.log(chalk.cyan('Following: ') + json.following);
        console.log(chalk.cyan('Public Repos: ') + json.public_repos);
        
        return json;
    } catch (error) {
        spinner.fail(`Error: ${error.message}`);
        throw error;
    }
}

// Fetch GitHub events
async function getEventData(username) {
    const url = `https://api.github.com/users/${username}/events`;
    const spinner = ora(`Fetching recent events for ${username}...`).start();
    
    try {
        const res = await fetch(url, getApiHeaders());
        if (!res.ok) {
            throw new Error(`Response status ${res.status}`);
        }
        const events = await res.json();
        spinner.succeed('Event data fetched successfully!');
        
        // Format and display events
        console.log(chalk.blue.bold('\nRecent Activities:'));
        events.slice(0, 10).forEach((event, index) => {
            const date = new Date(event.created_at).toLocaleDateString();
            let eventDescription = '';
            
            switch(event.type) {
                case 'PushEvent':
                    eventDescription = `Pushed to ${event.repo.name}`;
                    break;
                case 'CreateEvent':
                    eventDescription = `Created ${event.payload.ref_type} in ${event.repo.name}`;
                    break;
                case 'WatchEvent':
                    eventDescription = `Starred ${event.repo.name}`;
                    break;
                case 'IssuesEvent':
                    eventDescription = `${event.payload.action} issue in ${event.repo.name}`;
                    break;
                default:
                    eventDescription = `${event.type} in ${event.repo.name}`;
            }
            
            console.log(chalk.yellow(`${index + 1}. [${date}] `) + eventDescription);
        });
        
        return events;
    } catch (error) {
        spinner.fail(`Error: ${error.message}`);
        throw error;
    }
}

// API Routes with CORS headers
app.use((req, res, next) => {
    res.header('Access-Control-Allow-Origin', '*');
    res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept');
    next();
});

app.get("/api/profile/:username", async (req, res) => {
    try {
        const data = await getProfileData(req.params.username);
        res.json(data);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.get("/api/events/:username", async (req, res) => {
    try {
        const data = await getEventData(req.params.username);
        res.json(data);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// CLI Setup
const program = new Command();

program
    .version("1.0.0")
    .description("GitHub Detective - Explore GitHub profiles and activities")
    .option("-p, --profile", "Fetch GitHub profile data")
    .option("-e, --events", "Fetch GitHub events data")
    .option("-a, --all", "Fetch both profile and events data")
    .action(async (options) => {
        const { username } = await inquirer.prompt([
            {
                type: "input",
                name: "username",
                message: "Enter the GitHub username:",
                validate: input => input.length > 0 || "Username cannot be empty"
            }
        ]);

        try {
            if (options.all || (!options.profile && !options.events)) {
                await getProfileData(username);
                await getEventData(username);
            } else {
                if (options.profile) await getProfileData(username);
                if (options.events) await getEventData(username);
            }
        } catch (error) {
            console.error(chalk.red('Failed to fetch data'));
        }
    });

program.parse(process.argv);


const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(chalk.green(`\nServer running at http://localhost:${PORT}`));
});
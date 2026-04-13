# AI for App Development

## Building and deploying AI Apps that leverage agents, MCP and RAG ##

These instructions will guide you through configuring a GitHub Codespaces environment that you can use to do the labs. 

**1. Change your codespace's default timeout from 30 minutes to longer (60 for half-day sessions, 90 for deep dive sessions).**
To do this, when logged in to GitHub, go to https://github.com/settings/codespaces and scroll down on that page until you see the *Default idle timeout* section. Adjust the value as desired.

![Changing codespace idle timeout value](./images/31ai5.png?raw=true "Changing codespace idle timeout value")

**2. Click on the button below to start a new codespace from this repository.**

Click here ➡️  [![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/skillrepos/ai-apps_v3?quickstart=1)

**3. Then click on the option to create a new codespace.**

![Creating new codespace from button](./images/31ai1.png?raw=true "Creating new codespace from button")

This will run for a long time while it gets everything ready.

After the initial startup, it will run a script to setup the python environment and install needed python pieces. This will take several more minutes to run. It will look like this while this is running.

![Final prep](./images/31ai2.png?raw=true "Final prep")

The codespace is ready to use when you see a prompt like the one shown below in its terminal.

![Ready to use](./images/31ai3.png?raw=true "Ready to use")

**4. Set up your HuggingFace API token.**

The RAG agent uses HuggingFace's Inference API to generate LLM responses. You'll need a free API token:

A. Go to [https://huggingface.co](https://huggingface.co) and log in if you already have an account. If you need to create an account, click the *Sign Up* button or visit [https://huggingface.co/join](https://huggingface.co/join)

![HF login](./images/aia-3-19.png?raw=true "HF login")

<br>
   
B. Navigate to (https://huggingface.co/settings/tokens)[https://huggingface.co/settings/tokens].  Click on *+ Create new token*.

![Get token](./images/aia-3-20.png?raw=true "Get token")

<br>

C. Select **Write** for the token type and provide a name.

![Read token](./images/ae80.png?raw=true "Read token")

<br>
   
D. Click on the *Create token* button and copy the token (it starts with `hf_`). Save it somewhere.

![Save/copy token](./images/ae81.png?raw=true "Save/copy token")

<br>

E. For all runs of agents in the labs, make sure the token is set in your terminal before running the agent:

```bash
export HF_TOKEN="hf_your_token_here"
```

<br>

F. Alternatively, to make this permanent for your codespace session, add it to your shell profile:

```bash
echo 'export HF_TOKEN="hf_your_token_here"' >> ~/.bashrc
source ~/.bashrc
```

<br><br>

**5. Open up the *labs.md* file so you can follow along with the labs.**
You can either open it in a separate browser instance or open it in the codespace. 

![Opening labs](./images/v3appb29.png?raw=true "Opening labs")

**Now, you are ready for the labs!**



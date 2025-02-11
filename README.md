# ytsum
Quick project I made to summarize a YT video using GenAI

Future plans:

1. Optimize choice of models to enable edge computing/ pure CPU
2. Make a website demo (w/ DB for entries + API for genAI)

# setup
1. Install the appropriate CUDA packages if using CUDA
2. Make a conda environment
3. Install torch (test to make sure torch is using CUDA (e.g., torch.cuda.is_available()))
4. Install pip in conda (if not present), and then run `pip install -r requirements.txt`
5. Run: `python main.py`

# sample output
YouTube video link: [BREAKING NEWS: Trump Signs Raft Of New Executive Orders While Taking Questions From Reporters](https://www.youtube.com/watch?v=uE9FzUzAdg8')


## summary
```
1. The administration signs seven ambassador appointments, removes a training program, introduces a new FCA on corruption, and aims to strengthen U.S. economic policies.
2. Trump removes a training program, warns of new FCA, and emphasizes the importance of reducing border violations.
3. The new FCA targets corruption, with the government seeking funding and political maneuvering.
4. U.S. steel industry faces financial trouble, with tariffs on aluminum and steel expected to reduce production.
5. Trump takes questions from reporters, signaling his leadership and economic policy changes.
6. U.S. and China face tariffs, with U.S. steel companies arguing higher production and economic benefits.
7. U.S. steel industry faces challenges, with concerns about environmental damage and job losses.
8. tensions between U.S. and China over tariffs escalate, with both sides trying to avoid retaliation.
9. Hostages from Hamas are released, raising questions about political manipulation.
10. U.S. government actions include border control, drug control, and immigration reforms.
11. Executive orders aim to reduce border violations and combat fraud, with the government seeking funding.
12. U.S. policies focus on immigration reform and healthcare reform, with critics criticizing the process.
13. Biden meets with a group of individuals to address fraud and medicare issues, with concerns about layoffs.
14. Trump leads the administration, addressing reporters and taking questions from them.
15. The buyout of reporters raises questions about termination clauses and the government's obligations.
16. A global crisis involving confiscation, financial issues, and international cooperation is unfolding.
17. Trump's executive orders cause financial trouble, with transparency shifting and the penny disappearing.
```

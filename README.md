# UstaadAI - اردو بیالوجی استاد 🎓

> Gemma 4 powered Urdu Biology tutor for Pakistani Class 9 FBISE students

## Problem Statement
Pakistan mein 60%+ students Urdu medium mein padhte hain. Unke paas:
- English medium tutors nahi hain
- Mehange coaching centers affordable nahi hain  
- Internet slow ya expensive hai
- AI tools sirf English mein hain

**UstaadAI** in sab problems ka jawab hai.

## Live Demo
🤗 [Try UstaadAI on HuggingFace Spaces](https://huggingface.co/spaces/qasim-robotic/ustaadai-demo)

## Features
- ✅ Pure Urdu responses
- ✅ FBISE Class 9 Biology syllabus
- ✅ MCQ with correct answers
- ✅ Pakistani context examples
- ✅ Simple 4-part format (وضاحت، مثال، اہمیت، MCQ)
- ✅ Not limited to Biology only — can answer any questions in Urdu
## Architecture
Student Question (Urdu/Roman Urdu)
↓
Google Gemma 4 E2B
↓
Urdu System Prompt
↓
Structured Response (4 sections)
↓
Gradio UI

## Dataset
- 335 Urdu Biology Q&A examples
- FBISE Class 9 syllabus coverage
- Topics: Cell, Tissues, Biodiversity, Enzymes, Bioenergetics, Nutrition, Transport, Reproduction
- Available on Kaggle: [ustaadai-biology-dataset](https://www.kaggle.com/datasets/qasim26/ustaadai-biology-dataset)

## Setup Guide
```bash
# Install dependencies
pip install transformers torch torchvision gradio accelerate

# Run locally
python app.py
```

## Tech Stack
- **Model:** Google Gemma 4 E2B
- **Framework:** HuggingFace Transformers
- **UI:** Gradio
- **Training:** Kaggle T4 GPU
- **Deployment:** HuggingFace Spaces

## Future Roadmap
- [ ] Ollama integration for true offline use
- [ ] Physics aur Chemistry subjects add karna
- [ ] Android app banana
- [ ] Voice input/output Urdu mein
- [ ] More classes (10th, 11th, 12th)
- [ ] Fine-tuning with larger dataset (1000+ examples)

## Impact
- Target: 10M+ Urdu medium students in Pakistan
- Zero cost for students
- Works on basic smartphones
- No English knowledge required

## Made By
Muhammad Qasim | BSc CS Student | Virtual University Pakistan
- GitHub: [Qasim04026](https://github.com/Qasim04026)
- HuggingFace: [qasim-robotic](https://huggingface.co/qasim-robotic)

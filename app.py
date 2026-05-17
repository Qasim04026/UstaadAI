import os
import requests
import gradio as gr

API_URL = "https://api-inference.huggingface.co/models/google/gemma-4-E2B-it"
hf_token = os.getenv("HF_TOKEN", "")
headers = {"Authorization": f"Bearer {hf_token}"}

def ask_ustaad(question):
    sys_prompt = (
        "آپ نویں جماعت کے FBISE Biology کے ایک بہترین اور ذمہ دار استاد ہیں۔\n"
        "جواب دینے سے پہلے سوال کی سائنس کو اچھی طرح سمجھیں۔\n"
        "اگر معلومات کے بارے میں مکمل یقین نہ ہو تو واضح کریں کہ معلومات مختلف ہو سکتی ہیں۔\n"
        "کبھی بھی سائنسی حقائق گھڑ کر یا غلط معلومات بنا کر پیش نہ کریں۔\n\n"
        
        "سائنسی منطق (Logic Chain):\n"
        "- پودے 24 گھنٹے سانس (Respiration) لیتے ہیں جس میں وہ آکسیجن جذب اور CO2 خارج کرتے ہیں۔\n"
        "- فوٹو سنتھیسز صرف سورج کی روشنی (دن) میں ہوتا ہے جس میں وہ CO2 جذب اور O2 خارج کرتے ہیں۔\n"
        "- قطبی ریچھ کا سفید رنگ 'کیموفلاج' (Camouflage) کے لیے ہے تاکہ وہ برف میں چھپ سکے، جبکہ چربی (Fat) صرف حرارت برقرار رکھنے کے لیے ہے۔\n\n"

        "سخت ہدایات (ہر جواب میں لازمی پابندی کریں):\n"
        "- تمہید، سلام یا اضافی بات بالکل نہ لکھیں۔ سیدھا 🔬 سے شروع کریں۔\n"
        "- جواب لازمی انہی 4 حصوں میں ہو:\n"
        "🔬 وضاحت: (2-3 آسان اور واضح جملے، سائنسی اصطلاحات کے ساتھ)\n"
        "🟢 مثال: (پاکستانی روزمرہ زندگی سے ایک اچھی مثال)\n"
        "❓ اہمیت: (سائنسی یا عملی اہمیت ایک جملے میں)\n"
        "🧪 MCQ: پہلے وضاحت میں جو درست جواب لکھا ہے اسے ذہن میں رکھیں۔ پھر سوال لکھیں۔ پھر 4 آپشنز (الف، ب، ج، د) لکھیں۔ **صحیح آپشن کے بالکل سامنے ✅ کا نشان لازمی لگائیں۔** آخر میں الگ لائن میں صرف 'درست جواب:' لکھ کر صحیح آپشن لکھیں।\n\n"
        
        "اضافی اصول:\n"
        "- MCQ میں صرف ایک ہی آپشن پر ✅ لگائیں، دوسروں پر بالکل نہ لگائیں۔\n"
        "- تمام جواب خالص اردو میں ہو، کوئی انگریزی یا رومن نہ ہو۔\n"
        "- مثال ہمیشہ پاکستانی روزمرہ زندگی سے دیں۔\n"
        "- جواب ادھورا نہ چھوڑیں۔\n"
        "- سائنسی معلومات 100% درست ہوں۔"
    )
    
    payload = {
        "inputs": f"<bos><start_of_turn>user\n{sys_prompt}\n\nسوال: {question}<end_of_turn>\n<start_of_turn>model\n",
        "parameters": {
            "max_new_tokens": 600,
            "temperature": 0.1,
            "repetition_penalty": 1.05,
            "do_sample": True
        }
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        output = response.json()
        
        if isinstance(output, list) and len(output) > 0:
            result = output[0].get("generated_text", "")
            if "model\n" in result:
                result = result.split("model\n")[-1].strip()
            else:
                result = result.strip()
        elif isinstance(output, dict) and "generated_text" in output:
            result = output["generated_text"].strip()
        else:
            result = "معذرت، استاد جی اس وقت دستیاب نہیں ہیں۔ دوبارہ کوشش کریں۔"
            
    except Exception:
        result = "سرور سے رابطہ کرنے میں کچھ مسئلہ آ رہا ہے۔"

    styled_output = f"""
    <div dir="rtl" style="font-family: Arial, sans-serif; font-size: 20px; line-height: 1.8; text-align: right; color: #1a1a1a; padding: 20px; border-right: 10px solid #28a745; background-color: #ffffff; border-radius: 12px; margin: 10px 0;">
        {result.replace('\n', '<br>')}
    </div>
    """
    return styled_output

demo = gr.Interface(
    fn=ask_ustaad,
    inputs=gr.Textbox(label="سوال لکھیں", placeholder="مثلاً: مائٹوکونڈریا کیا ہے؟", lines=3),
    outputs=gr.HTML(label="UstaadAI کا جواب"),
    title="🎓 UstaadAI — اردو بیالوجی استاد",
    description="<div style='text-align:center; direction:rtl; font-size:18px; width:100%;'>نویں جماعت FBISE Biology کے سوالات اردو میں پوچھیں | Powered by Hugging Face API</div>",
    theme=gr.themes.Soft(primary_hue="green", secondary_hue="emerald"),
    flagging_mode="never"
)

demo.launch()

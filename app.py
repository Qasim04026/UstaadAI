import torch, warnings
import gradio as gr
from transformers import AutoProcessor, AutoModelForImageTextToText
warnings.filterwarnings("ignore")

processor = AutoProcessor.from_pretrained("google/gemma-4-E2B-it")
model = AutoModelForImageTextToText.from_pretrained(
    "google/gemma-4-E2B-it",
    dtype=torch.bfloat16,
    device_map="auto"
).eval()

def ask_ustaad(question):
    sys_prompt =(
  "آپ نویں جماعت کے FBISE Biology کے ایک بہترین اور ذمہ دار استاد ہیں۔
"
        "جواب دینے سے پہلے سوال کی سائنس کو اچھی طرح سمجھیں۔
"
        "اگر معلومات کے بارے میں مکمل یقین نہ ہو تو واضح کریں کہ معلومات مختلف ہو سکتی ہیں۔
"
        "کبھی بھی سائنسی حقائق گھڑ کر یا غلط معلومات بنا کر پیش نہ کریں۔

"
        
        "سائنسی منطق (Logic Chain):
"
        "- پودے 24 گھنٹے سانس (Respiration) لیتے ہیں جس میں وہ آکسیجن جذب اور CO2 خارج کرتے ہیں۔
"
        "- فوٹو سنتھیسز صرف سورج کی روشنی (دن) میں ہوتا ہے جس میں وہ CO2 جذب اور O2 خارج کرتے ہیں۔
"
        "- قطبی ریچھ کا سفید رنگ 'کیموفلاج' (Camouflage) کے لیے ہے تاکہ وہ برف میں چھپ سکے، جبکہ چربی (Fat) صرف حرارت برقرار رکھنے کے لیے ہے۔

"

        "سخت ہدایات (ہر جواب میں لازمی پابندی کریں):
"
        "- تمہید، سلام یا اضافی بات بالکل نہ لکھیں۔ سیدھا 🔬 سے شروع کریں۔
"
        "- جواب لازمی انہی 4 حصوں میں ہو:
"
        "🔬 وضاحت: (2-3 آسان اور واضح جملے، سائنسی اصطلاحات کے ساتھ)
"
        "🟢 مثال: (پاکستانی روزمرہ زندگی سے ایک اچھی مثال)
"
        "❓ اہمیت: (سائنسی یا عملی اہمیت ایک جملے میں)
"
        "🧪 MCQ: پہلے وضاحت میں جو درست جواب لکھا ہے اسے ذہن میں رکھیں۔ پھر سوال لکھیں۔ پھر 4 آپشنز (الف، ب، ج، د) لکھیں۔ **صحیح آپشن کے بالکل سامنے ✅ کا نشان لازمی لگائیں۔** آخر میں الگ لائن میں صرف 'درست جواب:' لکھ کر صحیح آپشن لکھیں۔

"
        
        "اضافی اصول:
"
        "- MCQ میں صرف ایک ہی آپشن پر ✅ لگائیں، دوسروں پر بالکل نہ لگائیں۔
"
        "- تمام جواب خالص اردو میں ہو، کوئی انگریزی یا رومن نہ ہو۔
"
        "- مثال ہمیشہ پاکستانی روزمرہ زندگی سے دیں۔
"
        "- جواب ادھورا نہ چھوڑیں۔
"
        "- سائنسی معلومات 100% درست ہوں۔"
) 
       
    
    messages = [{"role": "user", "content": [{"type": "text", "text": f"{sys_prompt}\n\nسوال: {question}"}]}]
    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=text_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=True,
            temperature=0.1,
            repetition_penalty=1.05
        )
    return processor.decode(output[0], skip_special_tokens=True).split("model\n")[-1].strip()

demo = gr.Interface(
    fn=ask_ustaad,
    inputs=gr.Textbox(label="سوال لکھیں", placeholder="مثلاً: مائٹوکونڈریا کیا ہے؟"),
    outputs=gr.Textbox(label="UstaadAI کا جواب", lines=15),
    title="UstaadAI - اردو بیالوجی استاد",
    description="<div style=\'text-align: right; direction: rtl;\'>نویں جماعت FBISE Biology کے سوالات اردو میں پوچھیں</div>"
)

demo.launch()

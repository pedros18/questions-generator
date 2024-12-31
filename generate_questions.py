import nltk
from transformers import T5ForConditionalGeneration, T5Tokenizer

#download necessary NLTK data files
nltk.download('punkt')

def generate_questions(text, num_questions=10):
    #loading pretrained model and tokenizer
    model_name = "valhalla/t5-base-qa-qg-hl"
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    #preprocess input text
    sentences = nltk.sent_tokenize(text)
    text_with_hl = " ".join(["<hl> " + sentence + " <hl>" for sentence in sentences])

    #encode input text
    input_text = f"generate questions: {text_with_hl} </s>"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    #generate questions
    outputs = model.generate(input_ids=input_ids, max_length=512, num_beams=10, num_return_sequences=num_questions)

    #decode and print questions
    questions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return questions

#example of artificial intelligence just pour essayer
text = """
Artificial intelligence (AI) is transforming various industries by enabling machines to perform tasks that traditionally required human intelligence. These tasks include understanding natural language, recognizing patterns, making decisions, and solving problems. AI technologies, such as machine learning and neural networks, have seen significant advancements in recent years, leading to breakthroughs in fields like healthcare, finance, and transportation. For instance, AI is being used to develop personalized treatment plans in medicine, predict stock market trends in finance, and create self-driving cars in transportation. Despite its benefits, AI also poses ethical and societal challenges, such as privacy concerns, job displacement, and decision-making transparency. As AI continues to evolve, it is crucial to address these challenges to ensure that its development and deployment are aligned with societal values and ethical principles. Researchers and policymakers are actively working on creating frameworks and guidelines to govern the use of AI technologies responsibly. By fostering collaboration between different stakeholders, it is possible to harness the potential of AI while mitigating its risks. The future of AI holds great promise, and with careful management, it can lead to significant improvements in various aspects of human life.
"""

questions = generate_questions(text, num_questions=10)
for i, question in enumerate(questions, 1):
    print(f"Question {i}: {question}")


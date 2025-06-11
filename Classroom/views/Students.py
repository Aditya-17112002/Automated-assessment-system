from django.contrib.auth import login, authenticate
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from ..decorators import student_required
from ..models import Classroom, Enrollment, Test, Question, Answer, testTaken
import datetime
from django.utils import timezone
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.db.models import Q

from sklearn.feature_extraction.text import TfidfTransformer
import nltk, string, numpy, math
from sklearn.feature_extraction.text import TfidfVectorizer


@login_required(login_url='login')
@student_required
def join_class(request):
	if request.method == "POST":
		code = request.POST['code']
		user = request.user 

		try:
			room = Classroom.objects.get(code=code)
		except Classroom.DoesNotExist:
			messages.warning(request, "There's no such Classroom")
			return redirect('join_class')

		if Enrollment.objects.filter(room=room, student=user).exists():
			messages.info(request, 'You Already Enrolled {}'.format(room))
		else: 
			Enrollment(room=room, student=user).save() 
			messages.success(request, '{} Class Enrolled'.format(room))

		return redirect('dashboard')

	return render(request, 'students/join_class.html')

import os
import json
import nltk
from dotenv import load_dotenv
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from google.cloud import vision
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer

# Load .env variables
load_dotenv()

# Set Google Cloud Credentials from .env
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')


@login_required(login_url='login')
@student_required
def attend_test(request, test_id):
    test = get_object_or_404(Test, id=test_id)

    if testTaken.objects.filter(test=test, student=request.user).exists():
        return redirect('review_test', test_id)

    qns = Question.objects.filter(test=test_id)
    return render(request, 'students/attend_test.html', {'qns': qns, 'test': test})



import os
import json
import nltk
import google.generativeai as genai
from dotenv import load_dotenv
from django.shortcuts import get_object_or_404, redirect, render
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from google.cloud import vision
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer

# Load environment variables from .env
load_dotenv()

# Configure Google Gemini API using key from .env
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

# Ensure NLTK data path is available
nltk.data.path.append('../../nltk_data/')


@login_required(login_url='login')
@student_required
def submit_test(request, test_id):
    # Initialize Google Vision API
    client = vision.ImageAnnotatorClient()

    test = get_object_or_404(Test, id=test_id)
    student = request.user 

    # Prevent duplicate submissions
    if testTaken.objects.filter(test=test, student=student).exists():
        return redirect('review_test', test_id)

    # Create testTaken entry
    tt = testTaken(test=test, student=student, actual_score=0, ml_score=0)
    tt.save()

    extracted_answers = {}

    # Function to normalize text for comparison
    def LemTokens(tokens):
        return [lemmer.lemmatize(token) for token in tokens]

    def LemNormalize(text):
        tokens = nltk.word_tokenize(text)
        words = [w.lower() for w in tokens if w.isalnum()]
        return LemTokens(words)

    def cos_similarity(textlist):
        tfidf = TfidfVec.fit_transform(textlist)
        return (tfidf * tfidf.T).toarray()

    # Ensure uploads directory exists
    upload_dir = "uploads/"
    os.makedirs(upload_dir, exist_ok=True)

    for q in Question.objects.filter(test=test_id):
        file = request.FILES.get(str(q.id))  # Get uploaded image

        if file:
            # Save the file to the uploads directory
            file_path = f"{upload_dir}{student.id}_{q.id}.jpg"
            path = default_storage.save(file_path, ContentFile(file.read()))

            with default_storage.open(path, "rb") as image_file:
                image = vision.Image(content=image_file.read())

            # Perform OCR using Google Vision API
            response = client.text_detection(image=image)
            texts = response.text_annotations
            extracted_text = texts[0].description if texts else "No text detected"

            # Store extracted text per question
            extracted_answers[q.id] = {
                "question": q.qn_text,
                "extracted_answer": extracted_text,
            }

            # Save extracted answer to the database
            answer = Answer(student=student, question=q, answer_text=extracted_text)

            # Get expected answer from database
            answer_key = q.key

            # **STEP 1: AI-Based Subjective Answer Evaluation**
            # **STEP 1: AI-Based Subjective Answer Evaluation**
            ai_prompt = f"Evaluate the following answer:\n\nQuestion: {q.qn_text}\nExpected Answer: {answer_key}\nStudent Answer: {extracted_text}\n\nProvide a score out of {q.max_score} and feedback."
            ai_response = model.generate_content(ai_prompt)
            ai_evaluation = ai_response.text  # This contains the AI-generated feedback

            # Extract AI-generated score (basic approach)
            score = q.max_score if "perfect" in ai_evaluation.lower() else (q.max_score * 0.7)

            # Save AI feedback and score in Answer model
            answer.ai_feedback = ai_evaluation
            answer.actual_score = int(round(score))
            answer.ml_score = int(round(score))
            answer.save()

            #  **STEP 2: TF-IDF Similarity Score**
            documents = [answer_key, extracted_text]
            documents = list(map(str, documents))
            lemmer = nltk.stem.WordNetLemmatizer()

            TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words=None)  # Remove 'english' stopwords
            tf_matrix = cos_similarity(documents)
            tfidfTran = TfidfTransformer(norm="l2")
            tfidfTran.fit(tf_matrix)
            tfidf_matrix = tfidfTran.transform(tf_matrix)
            cos_similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()
            tfidf_score = cos_similarity_matrix[0][1] * q.max_score

            # **STEP 3: Combine AI Score & TF-IDF Score**
            final_score = int(round((score + tfidf_score) / 2))

            # Save scores
            answer.actual_score = final_score
            answer.ml_score = final_score
            answer.save()
            tt.actual_score += answer.actual_score
            tt.ml_score += answer.ml_score

    # Save extracted answers in JSON
    json_path = f"uploads/extracted_answers_{student.id}_{test.id}.json"
    with open(json_path, "w") as json_file:
        json.dump(extracted_answers, json_file, indent=4)

    tt.save()
    return redirect('review_test', test_id)


@login_required(login_url='login')
@student_required
def review_test(request, test_id):
    test = get_object_or_404(Test, id=test_id)
    qns = Question.objects.filter(test=test_id)
    student = request.user 
    ans = []
    tot = 0
    act = 0  

    for q in qns:
        d = {"qns": q} 
        answer = Answer.objects.filter(student=student, question=q).first()  # Avoids 404
        
        if answer:
            act += answer.actual_score
            d["ans"] = answer
            d["ai_feedback"] = answer.ai_feedback if answer.ai_feedback else "No feedback available"
            d["nlp_score"] = answer.ml_score  
        else:
            d["ans"] = None
            d["ai_feedback"] = "No feedback available"
        
        tot += q.max_score 
        ans.append(d)


    mark = "{} / {}".format(act, tot)
    
    return render(request, 'students/review_test.html', { 
        'test': test, 
        'ans': ans, 
        'mark': mark 
    })


@login_required(login_url='login')
@student_required
def assigned_test(request, class_id):
	test = Test.objects.filter(belongs=class_id)

	tests = []
	for t in test:
		if(testTaken.objects.filter(test=t, student=request.user).exists()):
			continue 
		elif ( t.start_time == None or t.start_time < timezone.now()) and ( t.end_time == None or t.end_time > timezone.now()):
			t.status = "Assigned"
			tests.append(t)

	# Search
	search = request.GET.get('search')

	if search != "" and search is not None:
		tests = Test.objects.filter(belongs=class_id, name__icontains=search).order_by('-create_time')


	# paginator 
	paginator = Paginator(tests, 5)
	page = request.GET.get('page', 1)

	try:
		tests = paginator.page(page)
	except PageNotAnInteger:
		tests = paginator.page(1)
	except EmptyPage:
		tests = paginator.page(paginator.num_pages)

		
	room = get_object_or_404(Classroom, id=class_id)
	return render(request, 'classroom/view_class.html', {'tests' : tests, 'room' : room } )


@login_required(login_url='login')
@student_required
def missing_test(request, class_id):
	test = Test.objects.filter(belongs=class_id)

	tests = []
	for t in test:
		if(testTaken.objects.filter(test=t, student=request.user).exists()):
			continue 
		elif ( t.start_time == None or t.start_time < timezone.now()) and ( t.end_time == None or t.end_time > timezone.now()):
			t.status = "Assigned"
			tests.append(t)
		elif(t.start_time and t.start_time > timezone.now()): # test is not yet started
			t.status = "not"
		else:
			t.status = "late"
			tests.append(t)

	# Search
	search = request.GET.get('search')

	if search != "" and search is not None:
		tests = Test.objects.filter(belongs=class_id, name__icontains=search).order_by('-create_time')


	# paginator 
	paginator = Paginator(tests, 5)
	page = request.GET.get('page', 1)

	try:
		tests = paginator.page(page)
	except PageNotAnInteger:
		tests = paginator.page(1)
	except EmptyPage:
		tests = paginator.page(paginator.num_pages)

	room = get_object_or_404(Classroom, id=class_id)
	return render(request, 'classroom/view_class.html', {'tests' : tests, 'room' : room } )


@login_required(login_url='login')
@student_required
def done_test(request, class_id):
	taken = list(testTaken.objects.filter(student=request.user).values("test"))

	d = []
	for t in taken: 
		d.append( t['test'] )
	tests = Test.objects.filter(pk__in=d, belongs=class_id)

	# Search
	search = request.GET.get('search')

	if search != "" and search is not None:
		tests = Test.objects.filter(belongs=class_id, name__icontains=search).order_by('-create_time')


	# paginator 
	paginator = Paginator(tests, 5)
	page = request.GET.get('page', 1)

	try:
		tests = paginator.page(page)
	except PageNotAnInteger:
		tests = paginator.page(1)
	except EmptyPage:
		tests = paginator.page(paginator.num_pages)


	for t in tests:
		t.status = "done"
		
	room = get_object_or_404(Classroom, id=class_id)
	return render(request, 'classroom/view_class.html', {'tests' : tests, 'room' : room } )
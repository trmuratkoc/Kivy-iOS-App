import tkinter as tk
import webbrowser
from tkinter import messagebox, ttk, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import os
import webbrowser
from fpdf import FPDF
from PIL import Image, ImageTk

# Define the color palette
COLOR_BG = "#F7F9F2"
COLOR_PRIMARY = "#8DECB4"
COLOR_SECONDARY = "#41B06E"
COLOR_TERTIARY = "#3FA2F6"
COLOR_TEXT = "#000000"

# Define the percentage contribution for each score
contributions = {
    'EyeContact': [0, 2.5, 5],
    'SocialSmiling': [0, 2, 4],
    'SharedEnjoyment': [0, 2.5, 5],
    'UnderstandingSocialCues': [0, 2.5, 5],
    'Imitation': [0, 1.5, 3],
    'LanguageDevelopment': [0, 3, 6],
    'RepetitiveLanguage': [0, 1.5, 3],
    'UnusualProsody': [0, 0.5, 1],
    'BackAndForthConversation': [0, 2.5, 5],
    'UseOfGestures': [0, 1.5, 3],
    'ResponseToName': [0, 2.5, 5],
    'RepetitiveMotorMovements': [0, 2, 4],
    'InsistenceOnSameness': [0, 1.5, 3],
    'RestrictedInterests': [0, 2, 4],
    'UnusualSensoryInterests': [0, 0.5, 1],
    'SensorySensitivities': [0, 1.5, 3],
    'EpilepticSeizures': [0, 5],
    'FamilyHistoryOfAutism': [0, 2, 4],
    'PlaySkills': [0, 2, 4],
    'EmotionalExpression': [0, 1.5, 3],
    'PointingToObjects': [0, 2, 4],
    'InterestInRotatingObjects': [0, 1, 2],
    'CommunicationSkills': [0, 3, 6],
    'SocialInteraction': [0, 3, 6],
    'MotorSkills': [0, 2, 4],
    'CognitiveSkills': [0, 2.5, 5],
    'AdaptiveBehavior': [0, 2, 4]
}

descriptions = {
    'EyeContact': '0: Consistent Eye Contact: Maintains eye contact during interactions, showing engagement and social interest.\n1: Intermittent Eye Contact: Makes eye contact but often looks away. May appear distracted or uninterested.\n2: Limited or No Eye Contact: Avoids eye contact, often looking at objects or engaging in self-stimulatory behaviors.',
    'SocialSmiling': '0: Frequent and Spontaneous Smiling: Smiles readily in response to social interactions and positive experiences.\n1: Infrequent or Forced Smiling: Smiles less often or appears to smile in a forced or unnatural way.\n2: Rare or No Smiling: Rarely smiles, even in situations that typically elicit smiles in other children.',
    'SharedEnjoyment': '0: Active Shared Attention: Readily shares interest in toys, activities, or experiences with others through pointing, showing, or verbal communication.\n1: Limited Shared Attention: Sometimes shows interest in sharing experiences but may not initiate or sustain it consistently.\n2: Lack of Shared Attention: Rarely or never attempts to share attention or interest in objects or activities with others.',
    'UnderstandingSocialCues': '0: Perceptive to Social Cues: Easily understands and responds appropriately to non-verbal cues like facial expressions, tone of voice, and body language.\n1: Some Difficulty with Social Cues: Occasionally misinterprets or misses subtle social cues, leading to awkward or inappropriate responses.\n2: Significant Difficulty with Social Cues: Frequently misunderstands or ignores social cues, struggles to navigate social interactions, and may appear insensitive to others’ feelings.',
    'Imitation': '0: Spontaneous Imitation: Readily imitates actions, sounds, words, or gestures observed in others, demonstrating a natural ability to learn through observation.\n1: Delayed or Limited Imitation: Imitates others less frequently or with less accuracy than expected for their age. May require prompting or demonstration to replicate actions.\n2: Minimal or No Imitation: Rarely or never imitates others, even when encouraged or shown how. This can indicate challenges with social learning and communication development.',
    'LanguageDevelopment': '0: Age-Appropriate Language: Uses language skills (e.g., vocabulary, sentence structure, grammar) that are typical for their age.\n1: Mild Language Delay: Exhibits a slight delay in language development, using simpler sentences or having a smaller vocabulary than expected.\n2: Significant Language Delay: Demonstrates a notable delay in language development, possibly using single words or gestures instead of phrases, or having difficulty understanding spoken language.',
    'RepetitiveLanguage': '0: Rarely uses repetitive phrases or words out of context.\n1: Occasionally repeats phrases or words.\n2: Frequently repeats phrases or words, sometimes to the exclusion of other communication.',
    'UnusualProsody': '0: Speech has typical rhythm, tone, and pitch.\n1: Speech occasionally has atypical qualities (e.g., monotone, singsong voice).\n2: Speech consistently has atypical qualities.',
    'BackAndForthConversation': '0: Engages in age-appropriate back-and-forth conversations.\n1: Sometimes struggles to initiate or respond appropriately in conversations.\n2: Rarely or never engages in back-and-forth conversations.',
    'UseOfGestures': '0: Uses a variety of gestures to communicate (pointing, waving, etc.).\n1: Uses fewer gestures than expected or relies on a limited set of gestures.\n2: Rarely or never uses communicative gestures.',
    'ResponseToName': '0: Consistently responds to name when called, turning head and making eye contact.\n1: Responds to name intermittently or with a delay, sometimes requiring repetition or additional cues.\n2: Rarely or never responds to name, seems unaware when called, or appears overly focused on other stimuli.',
    'RepetitiveMotorMovements': '0: No or minimal repetitive motor movements.\n1: Occasional repetitive motor movements.\n2: Frequent and intense repetitive motor movements.',
    'InsistenceOnSameness': '0: Adapts easily to changes in routine or environment.\n1: Shows some distress with changes but can be redirected.\n2: Highly resistant to change and experiences significant distress.',
    'RestrictedInterests': '0: Wide range of interests.\n1: Shows some preference for a few specific interests.\n2: Intensely focused on a very limited number of interests.',
    'UnusualSensoryInterests': '0: No unusual sensory interests.\n1: Occasionally shows unusual interest in sensory aspects of objects.\n2: Frequently preoccupied with unusual sensory interests.',
    'SensorySensitivities': '0: Typical sensory responses.\n1: Occasionally over- or under-reactive to sensory input.\n2: Frequently over- or under-reactive to sensory input.',
    'EpilepticSeizures': '1: History of one or more epileptic seizures.',
    'FamilyHistoryOfAutism': '0: No known family members with autism.\n1: One or more first-degree relatives (parent, sibling) diagnosed with autism.\n2: More distant relatives (grandparent, aunt/uncle, cousin) diagnosed with autism.',
    'PlaySkills': '0: Engages in varied and imaginative play, often incorporating pretend scenarios and social interactions.\n1: Shows limited interest in imaginative play, prefers solitary play, or struggles with taking on roles in pretend play.\n2: Rarely engages in pretend play, prefers repetitive activities with objects, or avoids playing with other children.',
    'EmotionalExpression': '0: Expresses a wide range of emotions appropriately and in line with social context.\n1: Shows limited emotional expression or occasional difficulty matching emotions to situations.\n2: Exhibits flat affect (minimal emotional expression) or inappropriate emotional reactions.',
    'PointingToObjects': '0: Frequently points to objects to share interest with others.\n1: Sometimes points to objects but not consistently.\n2: Rarely or never points to objects to share interest.',
    'InterestInRotatingObjects': '0: No unusual interest in rotating objects.\n1: Occasionally shows interest in rotating objects.\n2: Frequently preoccupied with rotating objects.',
    'CommunicationSkills': '0: Communicates effectively with age-appropriate language.\n1: Has some difficulty with communication, occasionally unclear or inappropriate.\n2: Significant difficulty communicating, often unclear or inappropriate.',
    'SocialInteraction': '0: Interacts well with peers and adults.\n1: Some difficulty in social interactions, occasionally awkward or inappropriate.\n2: Significant difficulty in social interactions, often awkward or inappropriate.',
    'MotorSkills': '0: Age-appropriate motor skills.\n1: Some delay in motor skills.\n2: Significant delay in motor skills.',
    'CognitiveSkills': '0: Age-appropriate cognitive skills.\n1: Some difficulty with cognitive tasks.\n2: Significant difficulty with cognitive tasks.',
    'AdaptiveBehavior': '0: Age-Appropriate Skills: Demonstrates age-appropriate skills in daily living activities, communication, and social interaction.\n1: Mild Delays: Shows some delays in acquiring or using adaptive skills, but can generally function independently with minimal support.\n2: Significant Delays: Requires substantial support in daily living activities, communication, or social interaction.'
}

short_descriptions = {
    'EyeContact': 'Eye contact with others.',
    'SocialSmiling': 'Smiling in social interactions.',
    'SharedEnjoyment': 'Engaging in shared enjoyment activities.',
    'UnderstandingSocialCues': 'Understanding and responding to social cues.',
    'Imitation': 'Imitating others’ actions and sounds.',
    'LanguageDevelopment': 'Development of language skills.',
    'RepetitiveLanguage': 'Use of repetitive language.',
    'UnusualProsody': 'Unusual speech patterns.',
    'BackAndForthConversation': 'Engaging in back-and-forth conversations.',
    'UseOfGestures': 'Use of gestures for communication.',
    'ResponseToName': 'Response to being called by name.',
    'RepetitiveMotorMovements': 'Presence of repetitive motor movements.',
    'InsistenceOnSameness': 'Resistance to changes in routine.',
    'RestrictedInterests': 'Narrow range of interests.',
    'UnusualSensoryInterests': 'Unusual interest in sensory aspects.',
    'SensorySensitivities': 'Reactivity to sensory input.',
    'EpilepticSeizures': 'History of epileptic seizures.',
    'FamilyHistoryOfAutism': 'Family history of autism.',
    'PlaySkills': 'Ability to engage in play activities.',
    'EmotionalExpression': 'Range and appropriateness of emotional expressions.',
    'PointingToObjects': 'Pointing to objects.',
    'InterestInRotatingObjects': 'Interest in rotating objects.',
    'CommunicationSkills': 'Effective communication with language.',
    'SocialInteraction': 'Interactions with peers and adults.',
    'MotorSkills': 'Development of motor skills.',
    'CognitiveSkills': 'Ability to perform cognitive tasks.',
    'AdaptiveBehavior': 'Ability to adapt to daily life activities.'
}

suggestions = {
    (0, 20): "Low Risk: Your child's risk for autism appears low based on the current assessment. While there's no need for immediate concern, it's crucial to continue monitoring their development closely. Engage in activities that promote social interaction, communication, and play skills, such as reading together, singing songs, and playing turn-taking games. Pay close attention to their eye contact, response to name, use of gestures, and language development. Schedule regular check-ups with your pediatrician to track their progress and discuss any concerns you may have. Early intervention, even for mild delays, can be highly beneficial.",
    (20, 40): "Mildly Elevated Risk: While your child's risk for autism is mildly elevated, this does not necessarily mean they will develop ASD. It's recommended to schedule a developmental screening with your pediatrician or a child development specialist to assess your child's specific needs and strengths. Observe your child's play patterns, noting any preference for solitary or repetitive play. Engage in activities that encourage imitation, such as copying actions or sounds. Focus on fostering their communication skills by narrating daily activities, using simple language, and responding to their attempts to communicate. If your child experiences any persistent challenges with language, social interaction, or behavior, seek further evaluation and early intervention services.",
    (40, 60): "Moderately Elevated Risk: Your child's risk for autism is moderately elevated, and a comprehensive evaluation by a developmental specialist or autism specialist is strongly recommended. Early intervention can be crucial in addressing potential developmental delays or challenges. While waiting for the evaluation, continue to provide a supportive and structured environment for your child. Establish consistent routines, use visual schedules, and provide clear expectations. Engage in activities that encourage joint attention, such as following your child's lead and commenting on what they are interested in. Seek out resources and support groups for families affected by autism, such as online forums, parent training workshops, or local autism organizations. Early intervention services like speech therapy, occupational therapy, or Applied Behavior Analysis (ABA) can be beneficial in supporting your child's development.",
    (60, 80): "High Risk: Based on this assessment, your child is at high risk for autism. It's imperative to prioritize a comprehensive evaluation by a developmental specialist or autism specialist as soon as possible. Early intervention can make a significant difference in your child's development and long-term outcomes. In the meantime, focus on strengthening your bond with your child through affectionate interactions, reading together, and playing together. Provide a predictable and structured daily routine to help your child feel secure and reduce anxiety. Address any sensory sensitivities or behavioral challenges with the guidance of a professional. Explore early intervention programs that offer specialized therapies, such as speech therapy, occupational therapy, and ABA therapy, to address your child's specific needs and promote their development.",
    (80, 100): "Very High Risk: Given the results of this assessment, your child is at very high risk for autism. Prompt evaluation by a developmental specialist or autism specialist is critical. Early intervention is essential to maximize your child's potential and provide the necessary support. While awaiting evaluation, create a nurturing and understanding environment for your child. Focus on building communication skills through visual supports, sign language, or augmentative and alternative communication (AAC) devices if needed. Encourage social interaction through playgroups or social skills training programs. Seek out additional resources and support from autism organizations, parent groups, and online forums. Explore evidence-based interventions like ABA therapy, which has been shown to be effective for many children with autism. Remember that early intervention can significantly improve outcomes for children with autism, so taking action promptly is crucial."
}

def show_description(param):
    description = descriptions.get(param, "No description available.")
    messagebox.showinfo(param, description)

def calculate_risk():
    try:
        if not age_entry.get() or gender_var.get() == "Select Gender":
            messagebox.showerror("Input Error", "Please enter the age and gender of the child first.")
            return None

        total_risk = 0
        age = int(age_entry.get())
        gender = gender_var.get()
        
        # Gender adjustment factor based on scientific research
        gender_factor = 1.3 if gender == "Male" else 1.0  # Example factor: males generally have a higher risk
        
        # Age adjustment factors based on scientific research
        age_factors = {
            (0, 18): 1.2,   # Example factor: higher risk for younger age
            (19, 24): 1.1,
            (25, 36): 1.0,
            (37, 48): 0.9,  # Example factor: slightly lower risk as age increases
        }
        
        age_factor = next((factor for (start, end), factor in age_factors.items() if start <= age <= end), 1.0)
        
        inputs = [
            (input_vars['EyeContact'].get(), 'EyeContact'), (input_vars['SocialSmiling'].get(), 'SocialSmiling'),
            (input_vars['SharedEnjoyment'].get(), 'SharedEnjoyment'), (input_vars['UnderstandingSocialCues'].get(), 'UnderstandingSocialCues'),
            (input_vars['Imitation'].get(), 'Imitation'), (input_vars['LanguageDevelopment'].get(), 'LanguageDevelopment'),
            (input_vars['RepetitiveLanguage'].get(), 'RepetitiveLanguage'), (input_vars['UnusualProsody'].get(), 'UnusualProsody'),
            (input_vars['BackAndForthConversation'].get(), 'BackAndForthConversation'), (input_vars['UseOfGestures'].get(), 'UseOfGestures'),
            (input_vars['ResponseToName'].get(), 'ResponseToName'), (input_vars['RepetitiveMotorMovements'].get(), 'RepetitiveMotorMovements'),
            (input_vars['InsistenceOnSameness'].get(), 'InsistenceOnSameness'), (input_vars['RestrictedInterests'].get(), 'RestrictedInterests'),
            (input_vars['UnusualSensoryInterests'].get(), 'UnusualSensoryInterests'), (input_vars['SensorySensitivities'].get(), 'SensorySensitivities'),
            (input_vars['EpilepticSeizures'].get(), 'EpilepticSeizures'), (input_vars['FamilyHistoryOfAutism'].get(), 'FamilyHistoryOfAutism'),
            (input_vars['PlaySkills'].get(), 'PlaySkills'), (input_vars['EmotionalExpression'].get(), 'EmotionalExpression'),
            (input_vars['PointingToObjects'].get(), 'PointingToObjects'), (input_vars['InterestInRotatingObjects'].get(), 'InterestInRotatingObjects'),
            (input_vars['CommunicationSkills'].get(), 'CommunicationSkills'), (input_vars['SocialInteraction'].get(), 'SocialInteraction'),
            (input_vars['MotorSkills'].get(), 'MotorSkills'), (input_vars['CognitiveSkills'].get(), 'CognitiveSkills'),
            (input_vars['AdaptiveBehavior'].get(), 'AdaptiveBehavior')
        ]

        for score, param in inputs:
            if score:  # If there's a selection
                score = int(score)
                if param == 'EpilepticSeizures':
                    if score == 1:  # Only add if there's a history of seizures
                        total_risk += contributions[param][score]
                else:
                    total_risk += contributions[param][score]
        
        # Adjust risk based on gender and age factors
        total_risk *= gender_factor
        total_risk *= age_factor

        # Cap the risk percentage at 100%
        total_risk = min(total_risk, 100)

        # Update the result label with the calculated risk percentage
        result_label.config(text=f"Estimated Autism Risk: {total_risk:.2f}%")
        
        show_estimation_results_and_suggestions(total_risk)
        
        return total_risk
        
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def show_estimation_results_and_suggestions(risk):
    for widget in estimation_frame.winfo_children():
        widget.destroy()

    estimation_label = tk.Label(estimation_frame, text=f"Estimated Autism Risk: {risk:.2f}%", bg=COLOR_BG, fg=COLOR_TEXT, font=("Helvetica", 14, "bold"))
    estimation_label.pack(pady=5)

    suggestion_text = ""
    for (start, end), suggestion in suggestions.items():
        if start <= risk <= end:
            suggestion_text = suggestion
            break

    if risk <= 20:
        color = 'green'
    elif risk <= 40:
        color = 'yellow'
    elif risk <= 60:
        color = 'orange'
    elif risk <= 80:
        color = 'red'
    else:
        color = 'darkred'

    estimation_label.config(fg=color)

    suggestion_label = tk.Label(estimation_frame, text="Suggestions:", bg=COLOR_BG, fg=COLOR_TEXT, font=("Helvetica", 12, "bold"))
    suggestion_label.pack(pady=5)

    suggestion_message = tk.Message(estimation_frame, text=suggestion_text, bg=COLOR_BG, fg=COLOR_TEXT, font=("Helvetica", 12), width=500)
    suggestion_message.pack(pady=5)

def show_gauge_chart():
    for widget in gauge_chart_frame.winfo_children():
        widget.destroy()
    
    risk = calculate_risk()
    if risk is not None:
        gauge_fig, gauge_ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6, 3))
        gauge_ax.set_theta_direction(-1)
        gauge_ax.set_theta_offset(3.14)

        risk_normalized = risk / 100
        gauge_ax.barh(1, risk_normalized * 3.14, left=(1-risk_normalized) * 3.14, height=2, color="#4CAF50")

        gauge_ax.set_yticklabels([])
        gauge_ax.set_xticklabels([])

        # Adding legend and labels
        gauge_ax.text(3.14, 1.5, 'Risk Level', fontsize=12, horizontalalignment='center')
        gauge_ax.text(1.57, 1.5, f'{risk:.2f}%', fontsize=12, horizontalalignment='center', color='#FF0000')

        gauge_canvas = FigureCanvasTkAgg(gauge_fig, master=gauge_chart_frame)
        gauge_canvas.get_tk_widget().pack()
        gauge_canvas.draw()

def show_correlation_chart():
    # Simulated example for correlation chart
    np.random.seed(0)
    scores = np.random.rand(20)
    risks = np.random.rand(20)

    correlation_window = tk.Toplevel(root)
    correlation_window.title("Correlation Chart")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(scores, risks, c='blue')
    ax.set_xlabel('Parameter Scores')
    ax.set_ylabel('Risk Scores')
    ax.set_title('Correlation between Parameter Scores and Risk Scores')

    canvas = FigureCanvasTkAgg(fig, master=correlation_window)
    canvas.get_tk_widget().pack()
    canvas.draw()

def show_distribution_chart():
    # Simulated example for distribution chart
    parameters = [
        'EyeContact', 'SocialSmiling', 'SharedEnjoyment', 'UnderstandingSocialCues', 'Imitation',
        'LanguageDevelopment', 'RepetitiveLanguage', 'UnusualProsody', 'BackAndForthConversation', 
        'UseOfGestures', 'ResponseToName', 'RepetitiveMotorMovements', 'InsistenceOnSameness', 
        'RestrictedInterests', 'UnusualSensoryInterests', 'SensorySensitivities', 
        'EpilepticSeizures', 'FamilyHistoryOfAutism', 'PlaySkills', 'EmotionalExpression', 'PointingToObjects', 'InterestInRotatingObjects',
        'CommunicationSkills', 'SocialInteraction', 'MotorSkills', 'CognitiveSkills', 'AdaptiveBehavior'
    ]
    scores = [int(input_vars[param].get()) if input_vars[param].get() else 0 for param in parameters]

    # Filter out parameters with a score of 0 to avoid cluttering the chart
    filtered_params_scores = [(param, score) for param, score in zip(parameters, scores) if score > 0]
    if not filtered_params_scores:
        messagebox.showinfo("No Data", "No data available for distribution chart. Please select parameters with scores greater than 0.")
        return
    
    filtered_parameters, filtered_scores = zip(*filtered_params_scores)

    distribution_window = tk.Toplevel(root)
    distribution_window.title("Parameter Score Distribution")

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(filtered_parameters, filtered_scores, color='skyblue')
    ax.set_xlabel('Scores')
    ax.set_title('Distribution of Parameter Scores')

    # Add a legend
    ax.legend(filtered_parameters, loc="best")

    canvas = FigureCanvasTkAgg(fig, master=distribution_window)
    canvas.get_tk_widget().pack()
    canvas.draw()

def show_risk_breakdown():
    risk = calculate_risk()
    if risk is not None:
        inputs = [
            (input_vars['EyeContact'].get(), 'EyeContact'), (input_vars['SocialSmiling'].get(), 'SocialSmiling'), 
            (input_vars['SharedEnjoyment'].get(), 'SharedEnjoyment'), (input_vars['UnderstandingSocialCues'].get(), 'UnderstandingSocialCues'), 
            (input_vars['Imitation'].get(), 'Imitation'), (input_vars['LanguageDevelopment'].get(), 'LanguageDevelopment'), 
            (input_vars['RepetitiveLanguage'].get(), 'RepetitiveLanguage'), (input_vars['UnusualProsody'].get(), 'UnusualProsody'), 
            (input_vars['BackAndForthConversation'].get(), 'BackAndForthConversation'), (input_vars['UseOfGestures'].get(), 'UseOfGestures'), 
            (input_vars['ResponseToName'].get(), 'ResponseToName'), (input_vars['RepetitiveMotorMovements'].get(), 'RepetitiveMotorMovements'), 
            (input_vars['InsistenceOnSameness'].get(), 'InsistenceOnSameness'), (input_vars['RestrictedInterests'].get(), 'RestrictedInterests'), 
            (input_vars['UnusualSensoryInterests'].get(), 'UnusualSensoryInterests'), (input_vars['SensorySensitivities'].get(), 'SensorySensitivities'), 
            (input_vars['EpilepticSeizures'].get(), 'EpilepticSeizures'), (input_vars['FamilyHistoryOfAutism'].get(), 'FamilyHistoryOfAutism'), 
            (input_vars['PlaySkills'].get(), 'PlaySkills'), (input_vars['EmotionalExpression'].get(), 'EmotionalExpression'),
            (input_vars['PointingToObjects'].get(), 'PointingToObjects'), (input_vars['InterestInRotatingObjects'].get(), 'InterestInRotatingObjects'),
            (input_vars['CommunicationSkills'].get(), 'CommunicationSkills'), (input_vars['SocialInteraction'].get(), 'SocialInteraction'),
            (input_vars['MotorSkills'].get(), 'MotorSkills'), (input_vars['CognitiveSkills'].get(), 'CognitiveSkills'),
            (input_vars['AdaptiveBehavior'].get(), 'AdaptiveBehavior')
        ]

        contributions_data = []
        for score, param in inputs:
            if score and int(score) > 0:  # Only include scores of 1 or 2
                score = int(score)
                if param == 'EpilepticSeizures':
                    if score == 1:
                        contributions_data.append((param, contributions[param][score]))
                else:
                    contributions_data.append((param, contributions[param][score]))

        if contributions_data:
            risk_breakdown_window = tk.Toplevel(root)
            risk_breakdown_window.title("Risk Breakdown")

            fig, ax = plt.subplots(figsize=(8, 6))
            labels, sizes = zip(*contributions_data)
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
            ax.set_title('Contribution to Overall Risk')

            canvas = FigureCanvasTkAgg(fig, master=risk_breakdown_window)
            canvas.get_tk_widget().pack()
            canvas.draw()
        else:
            messagebox.showinfo("No Data", "No data available for risk breakdown. Please select parameters with scores of 1 or 2.")

def new_assessment():
    age_entry.delete(0, tk.END)
    gender_var.set("Select Gender")
    for var in input_vars.values():
        var.set(0)
    result_label.config(text="Estimated Autism Risk: N/A")
    for widget in estimation_frame.winfo_children():
        widget.destroy()
    for widget in gauge_chart_frame.winfo_children():
        widget.destroy()

def set_language(language):
    messagebox.showinfo("Language Selection", f"Language changed to {language}")

def reset_parameters():
    for var in input_vars.values():
        var.set(0)
    result_label.config(text="Estimated Autism Risk: N/A")
    for widget in estimation_frame.winfo_children():
        widget.destroy()
    for widget in gauge_chart_frame.winfo_children():
        widget.destroy()

def set_data_privacy():
    data_privacy_window = tk.Toplevel(root)
    data_privacy_window.title("Data Privacy and Security")
    data_privacy_window.geometry("400x300")
    data_privacy_window.configure(bg=COLOR_BG)

    tk.Label(data_privacy_window, text="Data Privacy and Security", bg=COLOR_BG, fg=COLOR_TEXT, font=("Helvetica", 14, "bold")).pack(pady=10)
    
    share_data_var = tk.BooleanVar(value=data_privacy_settings['share_data'])
    tk.Checkbutton(data_privacy_window, text="Allow anonymized data sharing for research purposes", variable=share_data_var, bg=COLOR_BG, fg=COLOR_TEXT, font=("Helvetica", 12)).pack(anchor='w', padx=20, pady=5)

    password_protect_var = tk.BooleanVar(value=data_privacy_settings['password_protect'])
    tk.Checkbutton(data_privacy_window, text="Enable password protection", variable=password_protect_var, bg=COLOR_BG, fg=COLOR_TEXT, font=("Helvetica", 12)).pack(anchor='w', padx=20, pady=5)
    
    tk.Button(data_privacy_window, text="Save Settings", bg=COLOR_PRIMARY, fg=COLOR_TEXT, font=("Helvetica", 12, "bold"), command=lambda: save_data_privacy_settings(share_data_var.get(), password_protect_var.get())).pack(pady=20)

def save_data_privacy_settings(share_data, password_protect):
    global data_privacy_settings
    data_privacy_settings['share_data'] = share_data
    data_privacy_settings['password_protect'] = password_protect
    messagebox.showinfo("Settings Saved", "Your data privacy settings have been saved.")

def set_display_options():
    display_options_window = tk.Toplevel(root)
    display_options_window.title("Display Options")
    display_options_window.geometry("400x300")
    display_options_window.configure(bg=COLOR_BG)

    tk.Label(display_options_window, text="Display Options", bg=COLOR_BG, fg=COLOR_TEXT, font=("Helvetica", 14, "bold")).pack(pady=10)
    
    chart_type_var = tk.StringVar(value=display_options_settings['chart_type'])
    tk.Label(display_options_window, text="Chart Type:", bg=COLOR_BG, fg=COLOR_TEXT, font=("Helvetica", 12)).pack(anchor='w', padx=20)
    tk.Radiobutton(display_options_window, text="Bar", variable=chart_type_var, value="Bar", bg=COLOR_BG, fg=COLOR_TEXT, font=("Helvetica", 12)).pack(anchor='w', padx=40)
    tk.Radiobutton(display_options_window, text="Pie", variable=chart_type_var, value="Pie", bg=COLOR_BG, fg=COLOR_TEXT, font=("Helvetica", 12)).pack(anchor='w', padx=40)

    color_scheme_var = tk.StringVar(value=display_options_settings['color_scheme'])
    tk.Label(display_options_window, text="Color Scheme:", bg=COLOR_BG, fg=COLOR_TEXT, font=("Helvetica", 12)).pack(anchor='w', padx=20)
    tk.OptionMenu(display_options_window, color_scheme_var, "Default", "High Contrast", "Color Blind Friendly").pack(anchor='w', padx=40, pady=5)

    font_size_var = tk.StringVar(value=display_options_settings['font_size'])
    tk.Label(display_options_window, text="Font Size:", bg=COLOR_BG, fg=COLOR_TEXT, font=("Helvetica", 12)).pack(anchor='w', padx=20)
    tk.OptionMenu(display_options_window, font_size_var, "Small", "Medium", "Large").pack(anchor='w', padx=40, pady=5)

    units_var = tk.StringVar(value=display_options_settings['units'])
    tk.Label(display_options_window, text="Units of Measurement:", bg=COLOR_BG, fg=COLOR_TEXT, font=("Helvetica", 12)).pack(anchor='w', padx=20)
    tk.OptionMenu(display_options_window, units_var, "Months", "Years").pack(anchor='w', padx=40, pady=5)

    tk.Button(display_options_window, text="Save Settings", bg=COLOR_PRIMARY, fg=COLOR_TEXT, font=("Helvetica", 12, "bold"), command=lambda: save_display_options(chart_type_var.get(), color_scheme_var.get(), font_size_var.get(), units_var.get())).pack(pady=20)

def save_display_options(chart_type, color_scheme, font_size, units):
    global display_options_settings
    display_options_settings['chart_type'] = chart_type
    display_options_settings['color_scheme'] = color_scheme
    display_options_settings['font_size'] = font_size
    display_options_settings['units'] = units
    messagebox.showinfo("Settings Saved", "Your display settings have been saved.")
    apply_display_settings()

def apply_display_settings():
    # Update the application's appearance based on display_options_settings
    # This should include updating the visual elements accordingly.
    # For simplicity, a placeholder message:
    messagebox.showinfo("Display Settings Applied", "Display settings have been applied. Restart the application to see the changes.")

def open_user_guide():
    try:
        webbrowser.open(r"C:/Users/MURATK\Desktop/our_tools/user_guide.pdf")
    except:
        messagebox.showerror("Error", "Unable to open user guide.")

def contact_support():
    webbrowser.open("mailto:info@datawisespectrum.com")

def save_assessment():
    file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
    if file_path:
        data = {
            'Age': [age_entry.get()],
            'Gender': [gender_var.get()],
        }
        for field, var in input_vars.items():
            data[field] = [var.get()]

        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)
        messagebox.showinfo("Save Assessment", "Assessment saved successfully.")

def open_assessment():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        df = pd.read_csv(file_path)
        age_entry.delete(0, tk.END)
        age_entry.insert(0, df['Age'][0])
        gender_var.set(df['Gender'][0])
        for field in fields:
            input_vars[field].set(df[field][0])
        messagebox.showinfo("Open Assessment", "Assessment loaded successfully.")

# Add other frames, widgets, and functionalities as needed

def export_results():
    file_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")])
    if file_path:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        pdf.cell(200, 10, txt="Autism Risk Assessment Report", ln=True, align='C')
        pdf.ln(10)

        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"Age: {age_entry.get()}", ln=True)
        pdf.cell(200, 10, txt=f"Gender: {gender_var.get()}", ln=True)
        pdf.cell(200, 10, txt=f"Estimated Autism Risk: {result_label.cget('text').split(': ')[1]}", ln=True)
        pdf.ln(10)

        pdf.cell(200, 10, txt="Assessment Details:", ln=True)
        pdf.set_font("Arial", size=10)
        for field, var in input_vars.items():
            pdf.cell(200, 10, txt=f"{field}: {var.get()}", ln=True)
        pdf.ln(10)

        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Suggestions:", ln=True)
        suggestion_text = ""
        for (start, end), suggestion in suggestions.items():
            if start <= float(result_label.cget('text').split(': ')[1][:-1]) <= end:
                suggestion_text = suggestion
                break
        pdf.multi_cell(0, 10, txt=suggestion_text)

        pdf.ln(10)
        pdf.cell(200, 10, txt="Sources:", ln=True)
        pdf.set_font("Arial", size=10)
        pdf.multi_cell(0, 10, txt="1. Source 1: https://doi.org/10.1176/appi.books.9780890425596\n2. Source 2: https://www.psychiatry.org/psychiatrists/practice/dsm\n3. Source 3: https://www.cdc.gov/mmwr/volumes/69/ss/ss6904a1.htm")

        pdf.output(file_path)
        messagebox.showinfo("Export Results", "Results exported successfully.")

def print_results():
    messagebox.showinfo("Print Results", "Print Results feature not implemented yet.")

from PIL import Image, ImageTk

# Create the landing page
def show_landing_page():
    landing_page = tk.Toplevel(root)
    landing_page.title("Welcome to NeuroGuard")
    landing_page.geometry("800x800")
    landing_page.configure(bg=COLOR_BG)

    # Load and display the logo
    logo_path = "C:/Users/MURATK/Desktop/bp_/1.png"  # Replace with the actual path to your logo
    logo_image = Image.open(logo_path)
    logo_image = logo_image.resize((100, 100), Image.LANCZOS)  # Resize the image as needed
    logo_photo = ImageTk.PhotoImage(logo_image)
    logo_label = tk.Label(landing_page, image=logo_photo, bg=COLOR_BG)
    logo_label.image = logo_photo  # Keep a reference to avoid garbage collection
    logo_label.pack(pady=10)

    welcome_label = tk.Label(
        landing_page,
        text="Welcome to NeuroGuard Autism Risk Predictor",
        bg=COLOR_BG,
        fg=COLOR_TEXT,
        font=("Helvetica", 24, "bold")
    )
    welcome_label.pack(pady=10)

    description_label = tk.Label(
        landing_page,
        text="NeuroGuard is an advanced, user-friendly tool designed to help parents and caregivers assess the risk of Autism Spectrum Disorder (ASD) in children aged 18 months to 4 years. Utilizing cutting-edge machine learning techniques and expert-validated behavioral parameters, NeuroGuard provides a personalized risk assessment to aid in early detection and intervention.",
        bg=COLOR_BG,
        fg=COLOR_TEXT,
        font=("Helvetica", 14),
        wraplength=600,
        justify="center"
    )
    description_label.pack(pady=20)

    how_it_works_label = tk.Label(
        landing_page,
        text="How It Works:",
        bg=COLOR_BG,
        fg=COLOR_TEXT,
        font=("Helvetica", 18, "bold"),
        wraplength=600,
        justify="left"
    )
    how_it_works_label.pack(pady=10)

    how_it_works_description = tk.Label(
        landing_page,
        text="1. Start Assessment: Begin by answering questions related to your child's eye contact, social interactions, language development, and other behavioral traits.\n"
             "2. Risk Calculation: Our tool calculates the risk percentage based on the responses and displays an estimated autism risk.\n"
             "3. In-Depth Analysis: Access various charts and visualizations, including gauge charts, correlation charts, and distribution charts, to understand the risk factors better.\n"
             "4. Machine Learning Insights: Benefit from the latest advancements in machine learning to get a comprehensive risk breakdown and performance statistics.",
        bg=COLOR_BG,
        fg=COLOR_TEXT,
        font=("Helvetica", 12),
        wraplength=600,
        justify="left"
    )
    how_it_works_description.pack(pady=10)

    start_button = ttk.Button(
        landing_page,
        text="Start Assessment",
        command=landing_page.destroy,
        style="TButton"
    )
    start_button.pack(pady=20)

    more_options_label = tk.Label(
        landing_page,
        text="More Options:",
        bg=COLOR_BG,
        fg=COLOR_TEXT,
        font=("Helvetica", 18, "bold"),
        wraplength=600,
        justify="left"
    )
    more_options_label.pack(pady=10)

    options_frame = tk.Frame(landing_page, bg=COLOR_BG)
    options_frame.pack(pady=10)

    data_privacy_button = ttk.Button(
        options_frame,
        text="Data Privacy Settings",
        command=set_data_privacy,
        style="TButton"
    )
    data_privacy_button.grid(row=0, column=0, padx=10, pady=10)

    display_options_button = ttk.Button(
        options_frame,
        text="Display Options",
        command=set_display_options,
        style="TButton"
    )
    display_options_button.grid(row=0, column=1, padx=10, pady=10)

    user_guide_button = ttk.Button(
        options_frame,
        text="User Guide",
        command=open_user_guide,
        style="TButton"
    )
    user_guide_button.grid(row=1, column=0, padx=10, pady=10)

    contact_support_button = ttk.Button(
        options_frame,
        text="Contact Support",
        command=contact_support,
        style="TButton"
    )
    contact_support_button.grid(row=1, column=1, padx=10, pady=10)

    contact_us_label = tk.Label(
        landing_page,
        text="Contact Us:",
        bg=COLOR_BG,
        fg=COLOR_TEXT,
        font=("Helvetica", 18, "bold"),
        wraplength=600,
        justify="left"
    )
    contact_us_label.pack(pady=10)

    contact_us_description = tk.Label(
        landing_page,
        text="For more information, support, or feedback, please reach out to us at info@datawisespectrum.com.",
        bg=COLOR_BG,
        fg=COLOR_TEXT,
        font=("Helvetica", 12),
        wraplength=600,
        justify="left"
    )
    contact_us_description.pack(pady=10)


root = tk.Tk()
root.title("NeuroGuard Autism Risk Predictor")
root.geometry("1200x900")
root.configure(bg=COLOR_BG)

# Global settings variables
data_privacy_settings = {
    'share_data': False,
    'password_protect': False
}

display_options_settings = {
    'chart_type': 'Bar',
    'color_scheme': 'Default',
    'font_size': 'Medium',
    'units': 'Months'
}

# Create a main frame
main_frame = tk.Frame(root, bg=COLOR_BG)
main_frame.pack(fill=tk.BOTH, expand=True)

# Modernize the menu bar
menubar = tk.Menu(root)
filemenu = tk.Menu(menubar, tearoff=0)
filemenu.add_command(label="New Assessment", command=new_assessment)
filemenu.add_command(label="Open Assessment", command=open_assessment)
filemenu.add_command(label="Save Assessment", command=save_assessment)
filemenu.add_command(label="Export Results", command=export_results)
filemenu.add_command(label="Print Results", command=print_results)
filemenu.add_separator()
filemenu.add_command(label="Exit", command=root.quit)
menubar.add_cascade(label="File", menu=filemenu)

viewmenu = tk.Menu(menubar, tearoff=0)
viewmenu.add_command(label="Toggle Full Screen", command=lambda: root.attributes("-fullscreen", not root.attributes("-fullscreen")))
menubar.add_cascade(label="View", menu=viewmenu)

toolsmenu = tk.Menu(menubar, tearoff=0)
toolsmenu.add_command(label="Calculate Risk", command=calculate_risk)
toolsmenu.add_command(label="Reset Parameters", command=reset_parameters)
menubar.add_cascade(label="Tools", menu=toolsmenu)

settingsmenu = tk.Menu(menubar, tearoff=0)
settingsmenu.add_command(label="Data Privacy", command=set_data_privacy)
settingsmenu.add_command(label="Display Options", command=set_display_options)
menubar.add_cascade(label="Settings", menu=settingsmenu)

reportmenu = tk.Menu(menubar, tearoff=0)
reportmenu.add_command(label="Generate Report", command=export_results)
menubar.add_cascade(label="Reports", menu=reportmenu)

helpmenu = tk.Menu(menubar, tearoff=0)
helpmenu.add_command(label="About", command=lambda: messagebox.showinfo("About", "Autism Risk Detection Tool\nVersion 1.0"))
helpmenu.add_command(label="User Guide", command=open_user_guide)
helpmenu.add_command(label="Contact Support", command=contact_support)
menubar.add_cascade(label="Help", menu=helpmenu)

root.config(menu=menubar)

# Left Frame for parameters and input fields
left_frame = tk.Frame(main_frame, bg=COLOR_BG)
left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

# Input fields for Age and Gender
age_gender_frame = tk.Frame(left_frame, bg=COLOR_BG)
age_gender_frame.pack(fill=tk.X, pady=5)

age_label = tk.Label(age_gender_frame, text="Child's Age (months):", bg=COLOR_BG, fg=COLOR_TEXT, font=("Helvetica", 12))
age_label.pack(side=tk.LEFT, padx=5)
age_entry = tk.Entry(age_gender_frame, font=("Helvetica", 12))
age_entry.pack(side=tk.LEFT, padx=5)

gender_label = tk.Label(age_gender_frame, text="Child's Gender:", bg=COLOR_BG, fg=COLOR_TEXT, font=("Helvetica", 12))
gender_label.pack(side=tk.LEFT, padx=5)
gender_var = tk.StringVar(value="Select Gender")
gender_dropdown = ttk.Combobox(age_gender_frame, textvariable=gender_var, values=["Male", "Female"], font=("Helvetica", 12))
gender_dropdown.pack(side=tk.LEFT, padx=5)

fields = [
    "EyeContact", "SocialSmiling", "SharedEnjoyment", "UnderstandingSocialCues", "Imitation",
    "LanguageDevelopment", "RepetitiveLanguage", "UnusualProsody", "BackAndForthConversation", 
    "UseOfGestures", "ResponseToName", "RepetitiveMotorMovements", "InsistenceOnSameness", 
    "RestrictedInterests", "UnusualSensoryInterests", "SensorySensitivities", 
    "EpilepticSeizures", "FamilyHistoryOfAutism", "PlaySkills", "EmotionalExpression", "PointingToObjects", "InterestInRotatingObjects",
    "CommunicationSkills", "SocialInteraction", "MotorSkills", "CognitiveSkills", "AdaptiveBehavior"
]

entries = []

# Define a style for the radio buttons
style = ttk.Style()
style.configure("TButton", background=COLOR_PRIMARY, foreground=COLOR_TEXT, font=("Helvetica", 12, "bold"))
style.map("TButton", background=[("active", COLOR_SECONDARY), ("disabled", "#cccccc")])


# Create variables to hold the selected values for each field
input_vars = {field: tk.IntVar(value=0) for field in fields}

# Function to toggle the visibility of subcategories
def toggle_subcategories(category_frame):
    for widget in category_frame.winfo_children():
        if isinstance(widget, tk.Frame):
            widget.pack_forget() if widget.winfo_ismapped() else widget.pack(fill=tk.X, pady=5)

# Function to create an accordion style frame
class Accordion(ttk.Frame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.columnconfigure(0, weight=1)

    def add_section(self, title, fields):
        section_frame = ttk.Frame(self)
        section_frame.grid(sticky='ew', pady=5)

        section_button = ttk.Button(self, text=title, command=lambda: self.toggle_section(section_frame))
        section_button.grid(sticky='ew')

        for field in fields:
            frame = tk.Frame(section_frame, bg=COLOR_BG)
            frame.pack(fill=tk.X, pady=5)

            label = tk.Label(frame, text=field, bg=COLOR_BG, fg=COLOR_TEXT, font=("Helvetica", 10, "bold"))
            label.pack(side=tk.LEFT)
            label.bind("<Button-1>", lambda e, f=field: show_description(f))

            short_desc = tk.Label(frame, text=f"({short_descriptions[field]})", bg=COLOR_BG, fg=COLOR_TEXT, font=("Helvetica", 9))
            short_desc.pack(side=tk.LEFT, padx=5)

            info_icon = tk.Label(frame, text="ℹ️", bg=COLOR_BG, fg=COLOR_TERTIARY, font=("Helvetica", 9))
            info_icon.pack(side=tk.LEFT)

            radio_frame = tk.Frame(frame, bg=COLOR_BG)
            radio_frame.pack(side=tk.RIGHT, padx=5)

            for val, text in enumerate(['0', '1', '2']):
                radio = ttk.Radiobutton(radio_frame, text=text, variable=input_vars[field], value=val, style="TRadiobutton")
                radio.pack(side=tk.LEFT, padx=5)

    def toggle_section(self, section_frame):
        if section_frame.winfo_viewable():
            section_frame.grid_remove()
        else:
            section_frame.grid()

# Define categories and fields
categories = {
    "Social Interaction": ["EyeContact", "SocialSmiling", "SharedEnjoyment", "UnderstandingSocialCues", "Imitation", "BackAndForthConversation", "PlaySkills", "SocialInteraction"],
    "Communication": ["LanguageDevelopment", "RepetitiveLanguage", "UnusualProsody", "UseOfGestures", "ResponseToName", "CommunicationSkills", "PointingToObjects"],
    "Behavior": ["RepetitiveMotorMovements", "InsistenceOnSameness", "RestrictedInterests", "InterestInRotatingObjects", "AdaptiveBehavior"],
    "Sensory": ["UnusualSensoryInterests", "SensorySensitivities", "EmotionalExpression"],
    "Medical": ["EpilepticSeizures", "FamilyHistoryOfAutism"],
    "Development": ["MotorSkills", "CognitiveSkills"]
}

accordion = Accordion(left_frame)
accordion.pack(fill=tk.X, pady=10)

for category, fields in categories.items():
    accordion.add_section(category, fields)


def load_and_train_model():
    predictor = AutismRiskPredictor()
    predictor.load_data('autism_data.csv')  # Ensure this CSV file is available in the same directory
    predictor.train_model()
    predictor.show_model_performance()

# Center frame for buttons and result label
center_frame = tk.Frame(main_frame, bg=COLOR_BG)
center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

calculate_button = ttk.Button(center_frame, text="Calculate Risk", command=calculate_risk, style="TButton")
calculate_button.pack(fill=tk.X, pady=5)

gauge_chart_button = ttk.Button(center_frame, text="Show Gauge Chart", command=show_gauge_chart, style="TButton")
gauge_chart_button.pack(fill=tk.X, pady=5)

correlation_chart_button = ttk.Button(center_frame, text="Show Correlation Chart", command=show_correlation_chart, style="TButton")
correlation_chart_button.pack(fill=tk.X, pady=5)

distribution_chart_button = ttk.Button(center_frame, text="Show Distribution Chart", command=show_distribution_chart, style="TButton")
distribution_chart_button.pack(fill=tk.X, pady=5)

risk_breakdown_button = ttk.Button(center_frame, text="Show Risk Breakdown", command=show_risk_breakdown, style="TButton")
risk_breakdown_button.pack(fill=tk.X, pady=5)

train_model_button = ttk.Button(center_frame, text="Train Model", command=load_and_train_model, style="TButton")
train_model_button.pack(fill=tk.X, pady=5)

result_label = tk.Label(center_frame, text="Estimated Autism Risk: N/A", bg=COLOR_BG, fg=COLOR_TEXT, font=("Helvetica", 14, "bold"))
result_label.pack(fill=tk.X, pady=10)

footer = tk.Label(center_frame, text="© 2024 NeuroGuard Autism Risk Predictor", bg=COLOR_PRIMARY, fg=COLOR_TEXT, font=("Helvetica", 10))
footer.pack(fill=tk.X, pady=10)

# Right Frame for charts and estimation results
right_frame = tk.Frame(main_frame, bg=COLOR_BG)
right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

# Gauge Chart placeholder
gauge_chart_frame = tk.Frame(right_frame, bg=COLOR_BG, relief=tk.SUNKEN, borderwidth=2)
gauge_chart_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
gauge_chart_label = tk.Label(gauge_chart_frame, text="Gauge Chart Placeholder", bg=COLOR_BG, fg=COLOR_TEXT, font=("Helvetica", 12))
gauge_chart_label.pack(pady=5)

# Estimation Results and Suggestions placeholder
estimation_frame = tk.Frame(right_frame, bg=COLOR_BG, relief=tk.SUNKEN, borderwidth=2)
estimation_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
estimation_label = tk.Label(estimation_frame, text="Estimation Results and Suggestions Placeholder", bg=COLOR_BG, fg=COLOR_TEXT, font=("Helvetica", 12))
estimation_label.pack(pady=5)

show_landing_page()

# Machine Learning Module
class AutismRiskPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def load_data(self, filename):
        self.data = pd.read_csv(filename)
        self.X = self.data.drop(columns=['ASD'])
        self.y = self.data['ASD']

    def train_model(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)
        self.model.fit(X_train, y_train)
        self.y_pred = self.model.predict(X_test)
        self.y_test = y_test
        print("Model trained with accuracy:", accuracy_score(y_test, self.y_pred))

    def show_model_performance(self):
        report = classification_report(self.y_test, self.y_pred, target_names=['No ASD', 'ASD'])
        cm = confusion_matrix(self.y_test, self.y_pred)
        accuracy = accuracy_score(self.y_test, self.y_pred)
        cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No ASD', 'ASD'])

        # Create a new window for displaying the performance
        performance_window = tk.Toplevel(root)
        performance_window.title("Model Performance")
        performance_window.geometry("800x600")
        
        # Add text explanation
        explanation = (
            f"Accuracy: {accuracy:.2f}\n\n"
            "Classification Report:\n"
            f"{report}\n\n"
            "Explanation:\n"
            "Accuracy is the ratio of correctly predicted instances to the total instances.\n"
            "Precision is the ratio of correctly predicted positive observations to the total predicted positives.\n"
            "Recall (Sensitivity) is the ratio of correctly predicted positive observations to all the observations in actual class.\n"
            "F1 Score is the weighted average of Precision and Recall.\n"
        )
        
        explanation_label = tk.Label(performance_window, text=explanation, bg=COLOR_BG, fg=COLOR_TEXT, font=("Helvetica", 12), justify="left")
        explanation_label.pack(pady=10, padx=10, anchor='w')

        # Display the confusion matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        cm_display.plot(ax=ax, cmap='Blues')
        
        canvas = FigureCanvasTkAgg(fig, master=performance_window)
        canvas.get_tk_widget().pack()
        canvas.draw()

        # Additional Seaborn heatmap for the confusion matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['No ASD', 'ASD'], yticklabels=['No ASD', 'ASD'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')

        canvas2 = FigureCanvasTkAgg(fig, master=performance_window)
        canvas2.get_tk_widget().pack()
        canvas2.draw()

root.mainloop()


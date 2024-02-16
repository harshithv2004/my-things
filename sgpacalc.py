# Define the grade points and credit points according to VTU's 2022 scheme
grade_points = {
    'S': 10, 'A': 9, 'B': 8, 'C': 7, 'D': 6, 'E': 5, 'F': 0,
}

# Initialize variables for total grade points and total credits
total_grade_points = 0
total_credits = 0

# Ask the user for the number of subjects
num_subjects = int(input("Enter the number of subjects: "))

# Iterate through each subject and get user input for marks and credits
for i in range(num_subjects):
    subject_code = input(f"Enter subject code for Subject {i + 1}: ")
    credits = int(input(f"Enter credits for {subject_code}: "))
    marks = int(input(f"Enter marks for {subject_code}: "))
    
    if marks >= 90:
        grade = 'S'
    elif 80 <= marks < 90:
        grade = 'A'
    elif 70 <= marks < 80:
        grade = 'B'
    elif 60 <= marks < 70:
        grade = 'C'
    elif 45 <= marks < 60:
        grade = 'D'
    else:
        grade = 'F'

    # Calculate grade points for the subject
    grade_point = grade_points[grade]

    # Calculate total grade points and total credits
    total_grade_points += grade_point * credits
    total_credits += credits

# Calculate SGPA
sgpa = total_grade_points / total_credits

print(f"Your SGPA is: {sgpa:.2f}")

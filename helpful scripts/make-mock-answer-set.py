import numpy as np
import pandas as pd
import random
import string
import sys

def load_answer_key(filename):
    """Load answer key from comma-delimited text file"""
    with open(filename, 'r') as f:
        key = [answer.strip() for answer in f.read().split(',')]
    return key

def get_possible_answers(key):
    """Extract unique possible answers from the key"""
    possible_answers = sorted(list(set(key)))
    return possible_answers

def generate_fake_names(num_students):
    """Generate random student names and IDs"""
    first_names = ['James', 'Mary', 'Robert', 'Patricia', 'Michael', 'Jennifer', 'William', 'Linda',
                   'David', 'Barbara', 'Richard', 'Elizabeth', 'Joseph', 'Susan', 'Thomas', 'Jessica',
                   'Charles', 'Sarah', 'Christopher', 'Karen', 'Daniel', 'Nancy', 'Matthew', 'Lisa',
                   'Anthony', 'Betty', 'Donald', 'Margaret', 'Mark', 'Sandra', 'Steven', 'Ashley',
                   'Paul', 'Kimberly', 'Andrew', 'Emily', 'Joshua', 'Donna', 'Kenneth', 'Michelle',
                   'Kevin', 'Dorothy', 'Brian', 'Carol', 'George', 'Amanda', 'Edward', 'Melissa',
                   'Ronald', 'Deborah', 'Timothy', 'Stephanie', 'Jason', 'Rebecca', 'Jeffrey',
                   'Alan', 'Omar', 'Jared', 'Zara', 'Quinn', 'Pearl', 'Blair', 'Fiona', 'Carla',
                   'Jenna', 'Tyler', 'Owen', 'Rosie', 'Clara', 'Nina', 'Brent', 'Diana', 'Wade',
                   'Molly', 'Anna', 'Zack', 'Mason', 'Derek', 'Adam', 'Abby', 'Yara', 'Nate',
                   'Avery', 'Jack', 'Zoey', 'Sharon']

    last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones',
                  'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez',
                  'Hernandez', 'Lopez', 'Gonzalez', 'Wilson',
                  'Anderson', 'Thomas', 'Taylor', 'Moore', 'Jackson',
                  'Martin', 'Lee', 'Perez', 'Thompson', 'White',
                  'Harris', 'Sanchez', 'Clark', 'Ramirez', 'Lewis',
                  'Robinson', 'Young', 'Allen', 'King', 'Wright', 'Scott',
                  'Torres', 'Peterson', 'Phillips', 'Campbell', 'Parker',
                  'Evans', 'Edwards', 'Collins', 'Reyes', 'Stewart',
                  'Morris', 'Morales', 'Murphy', 'Cook', 'Rogers',
                  'Morgan', 'Cooper', 'Reed', 'Bell', 'Berg', 'Kumar',
                  'Rahman', 'Klein', 'Novak', 'Silva', 'Popescu', 'Barros',
                  'Tanaka', 'Anderson', 'Li', 'Voinova', 'Costa', 'Flores',
                  'Wilson', 'Kowalski', 'Blanco', 'Bishop', 'Desai',
                  'Ford', 'Hassan', 'Banerjee', 'Andersson', 'Espinoza',
                  'Davis', 'Ma', 'Ahmed', 'Patel', 'Duran', 'Ali', 'OBrien',
                  'Ghosh', 'Singh', 'Bhat', 'Farah', 'Saito', 'Cohen',
                  'Reddy', 'Bianchi', 'Black', 'Dubois', 'Dong', 'DelaCruz',
                  'Lima', 'Chen', 'Kobayashi', 'Laurent', 'Carter', 'Adams',
                  'Araya', 'Baker', 'Rossi', 'Bae', 'Le', 'Santos', 'Mehta',
                  'Dominguez', 'Hall', 'Delgado', 'Lin', 'Qureshi',
                  'Becker', 'Rivera', 'Fernandez', 'Aung', 'Petrov',
                  'Hernandez', 'Burke', 'Green', 'Diaz', 'Suzuki',
                  'Machado', 'Bianchi', 'Chang', 'Pereira', 'El-Sayed',
                  'Fischer', 'Gupta', 'Ortega', 'Bennett', 'Moreno',
                  'Campbell', 'Kruger', 'Kimura', 'Nunez', 'Choi',
                  'Kuznetsov', 'Cruz']

    # Generate unique student IDs
    student_ids = set()
    while len(student_ids) < num_students:
        first_digit = random.randint(1, 9)  #don't start student ID with zero
        remaining_digits = ''.join(random.choices(string.digits, k=9))
        student_id = str(first_digit) + remaining_digits
        student_ids.add(student_id)

    
    student_ids = sorted(list(student_ids))
    
    # Generate random names
    names = []
    for i in range(num_students):
        first = random.choice(first_names)
        last = random.choice(last_names)
        names.append((student_ids[i], first, last))
    
    return names

def generate_fake_answers(correct_answers, num_students=100):
    """Generate fake student answers with distribution from 20-100%"""
    possible_answers = get_possible_answers(correct_answers)
    num_questions = len(correct_answers)
    
    # Create accuracy scores with median around 75%, tail toward 20%
    accuracy_scores = np.concatenate([
        [1.00],                                           # one perfect student
        [0.20],                                           # one struggling student
        np.random.beta(3.5, 2.5, num_students - 2) * 0.80 + 0.20
    ])
    
    np.random.shuffle(accuracy_scores)
    
    # Create the fake answer matrix
    fake_answers = []
    
    for i in range(num_students):
        num_correct = round(accuracy_scores[i] * num_questions)
        
        # Create array of student answers
        student_answers = [None] * num_questions
        
        # Choose which questions they get right
        correct_indices = np.random.choice(num_questions, num_correct, replace=False)
        for idx in correct_indices:
            student_answers[idx] = correct_answers[idx]
        
        # Fill in wrong answers for incorrect questions
        wrong_indices = [j for j in range(num_questions) if student_answers[j] is None]
        for j in wrong_indices:
            # Decide: skip answer (2%), multi-answer (2%), or single wrong answer (96%)
            rand = random.random()
            
            if rand < 0.02:
                # Skip the answer (leave blank)
                student_answers[j] = ''
            elif rand < 0.04:
                # Multi-answer: pick 2 different answers
                multi_answers = random.sample(possible_answers, k=2)
                student_answers[j] = ','.join(multi_answers)
            else:
                # Single wrong answer
                wrong_options = [ans for ans in possible_answers if ans != correct_answers[j]]
                student_answers[j] = np.random.choice(wrong_options)
        
        fake_answers.append(student_answers)
    
    # Convert to numpy array (use object dtype to handle varying string lengths)
    fake_answers = np.array(fake_answers, dtype=object)
    
    # Calculate actual scores (only count single correct answers, not multi or skipped)
    actual_scores = np.sum(fake_answers == np.array(correct_answers), axis=1) / num_questions
    
    return fake_answers, actual_scores

def main():
    # Check for command line argument
    if len(sys.argv) < 2:
        print("Usage: python script.py key.txt")
        print("Expected: A comma-delimited text file with answer key")
        sys.exit(1)
    
    key_file = sys.argv[1]
    
    # Load the answer key
    try:
        correct_answers = load_answer_key(key_file)
        print(f"Loaded answer key with {len(correct_answers)} questions")
        print(f"Possible answers: {get_possible_answers(correct_answers)}")
    except FileNotFoundError:
        print(f"Error: Could not find file '{key_file}'")
        sys.exit(1)
    
    np.random.seed(42)
    
    # Generate fake data
    num_students = 100
    fake_answers, actual_scores = generate_fake_answers(correct_answers, num_students)
    fake_names = generate_fake_names(num_students)
    
    # Create dataframe
    df = pd.DataFrame(fake_names, columns=['StudentID', 'FirstName', 'LastName'])
    
    # Add scores, rounded to 1 decimal place
    df['Score'] = (actual_scores * 100).round(1)
    
    # Add the answers as columns
    answers_df = pd.DataFrame(fake_answers, columns=[f"Q{i+1}" for i in range(len(correct_answers))])
    df = pd.concat([df, answers_df], axis=1)

    # Write to CSV with answer key as first row
    output_file = 'fake_scantron_answers.csv'
    
    # Create a row for the answer key
    key_row = pd.DataFrame([['ANSWER_KEY', '', '', ''] + correct_answers], 
                           columns=df.columns)
                           
    # Concatenate key row with student data
    df_with_key = pd.concat([key_row, df], ignore_index=True)
    df_with_key.to_csv(output_file, index=False)
    
    # Print summary
    print(f"\nGenerated {num_students} fake students")
    print(f"Score distribution:")
    print(f"  Min: {actual_scores.min() * 100:.1f}%")
    print(f"  Max: {actual_scores.max() * 100:.1f}%")
    print(f"  Median: {np.median(actual_scores) * 100:.1f}%")
    print(f"  Mean: {actual_scores.mean() * 100:.1f}%")
    print(f"\nScores above 90%: {sum(actual_scores > 0.90)}")
    print(f"\nSaved to '{output_file}'")
    print(f"\nFirst 5 rows:")
    print(df.head())

if __name__ == '__main__':
    main()

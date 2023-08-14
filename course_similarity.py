from argparse import ArgumentParser
import os
from termcolor import colored as col
import json
import numpy as np
import spacy
import time


def time_func(func):
  def wrapper(*args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    print(col(f"Time taken for {func.__name__}: {end - start:.2f}s", "green"))
    return result
  return wrapper


@time_func
def load_course_data(path):
  with open(path) as file:
    return { k: v["description"] for (k, v) in json.load(file).items() }


class Status:
  def __init__(self, name: str) -> None:
    self.name = name
    self.start = 0
    self.end = 0

  def __enter__(self) -> None:
    # Print name
    print(f"{self.name}...", end=" ")
    self.start = time.time()
  
  def __exit__(self, exc_type, exc_value, traceback) -> None:
    # Print time taken
    self.end = time.time()
    print(col(f"{self.end - self.start:.2f}s", "light_green"), end=" ")

    # Print status
    if exc_type is None:
      print(col("Good", "green"))
    else:
      print(col("Bad", "red"))
      print(exc_value)


@time_func
def main():
  # Parse command line arguments
  parser = ArgumentParser()
  parser.add_argument("id", type=str, help="Course ID to find similar courses for")
  parser.add_argument("data", type=str, help="Path to course data JSON file")
  parser.add_argument("n", type=int, help="Number of similar courses to find")
  args = parser.parse_args()

  # Load JSON with course data and descriptions
  with Status("Loading course data"):
    course_data = load_course_data(args.data)

  # Validate input course ID
  if args.id not in course_data:
    raise ValueError("Invalid course ID")

  # Load spacy English model
  with Status("Loading spacy model"):
    try:
      # Try to load spacy English model from disk
      nlp = spacy.load("data/en_core_web_sm")

    except Exception as e:
      # Download spacy English model
      nlp = spacy.load("en_core_web_sm")
      
      # Save spacy English model to disk
      os.makedirs("data", exist_ok=True)
      nlp.to_disk("data/en_core_web_sm")

  # Create vector database with course descriptions
  with Status("Creating vector database"):
    vector_database = {}
    for (course_id, course_description) in course_data.items():
      # Process course description using spacy
      doc = nlp(course_description)
      
      # Average word embeddings to get sentence embedding
      sentence_vector = np.mean([tok.vector for tok in doc], axis=0)
      
      # Store sentence embedding
      vector_database[course_id] = sentence_vector

    # Process input course description using spacy
    input_course_description = course_data[args.id]
    input_doc = nlp(input_course_description)

    # Average word embeddings to get sentence embedding for input course
    input_vector = np.mean([token.vector for token in input_doc], axis=0)

  # Calculate similarity between input course and all other courses
  with Status("Calculating similarities"):
    similarities = {}
    
    for (course_id, course_vector) in vector_database.items():
      # Exclude input course ID from similarity calculation as it will always be most similar to itself
      if course_id == args.id:
        continue
      
      # Calculate similarity between input course and current course
      similarity_score = np.dot(input_vector, course_vector) / (np.linalg.norm(input_vector) * np.linalg.norm(course_vector))

      # Store similarity score
      similarities[course_id] = similarity_score

    # Sort courses by similarity score in descending order
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

  #
  # Print results
  #

  # Print header
  print()
  print("*" * 80)
  print()
  print(col("Results:", "magenta", attrs=["bold"]))
  print()
  print(col("Input Course ID:",          "cyan" ), col(args.id,                  "light_cyan"))
  print(col("Input Course Description:", "cyan" ), col(input_course_description, "light_cyan"))
  print()

  # Print N most similar courses
  for n in range(args.n):
    # Get course ID and similarity score
    most_similar_course_id = sorted_similarities[n][0]
    most_similar_course_score = sorted_similarities[n][1]
    most_similar_course_description = course_data[most_similar_course_id]

    print(col("Most Similar Course ID:",          "green"), col(most_similar_course_id,             "light_green"))
    print(col("Most Similar Course Score:",       "green"), col(f"{most_similar_course_score:.2f}", "light_green"))
    print(col("Most Similar Course Description:", "green"), col(most_similar_course_description,    "light_green"))
    print()
  
  # Print footer
  print("*" * 80)
  print()


if __name__ == "__main__":
  main()

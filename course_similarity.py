# python3 -m spacy download en_core_web_sm
# python3 -m spacy download en_core_web_md
# python3 -m spacy download en_core_web_lg

from argparse import ArgumentParser
from functools import lru_cache
from typing import Any, Dict, List
import json
import numpy as np

from utilities import col, ScopeTimer

with ScopeTimer("Loading model"):
  import spacy, en_core_web_sm
  spacy.prefer_gpu()
  NLP = en_core_web_sm.load()


class CourseSimilarityVectorDatabase:
  def __init__(self, course_data_path: str):
    # Load course data
    with open(course_data_path) as file:
      self.course_data = json.load(file)
    
    # Build vector database
    self.vector_database = {
      course_id: self._get_course_vector(course_info)
      for (course_id, course_info) in self.course_data.items()
    }


  @staticmethod
  def _get_course_vector(course_info: Dict) -> Any:
    # Have to use inner function to cache results instead because the course info is a dict, which is not hashable
    @lru_cache(maxsize=1024)
    def _func(tokens: str) -> Any:
      return np.mean([tok.vector for tok in NLP(tokens)], axis=0)
    
    # Return the course vector
    return _func(course_info["title"] + course_info["description"])


  @lru_cache(maxsize=1024)
  def find_most_similar_courses(self, input_course_id: str) -> List[Dict]:
    # Validate input course ID
    if input_course_id not in self.course_data:
      raise ValueError("Invalid course ID")

    # Calculate vector for input course description
    input_vector = self._get_course_vector(self.course_data[input_course_id])

    # Calculate similarity between input course and all other courses
    similarities = {}
    for (course_id, course_vector) in self.vector_database.items():
      # Exclude input course ID from similarity calculation as it will always be most similar to itself
      if course_id == input_course_id:
        continue
      
      # Calculate similarity between input course and current course
      similarity_score = np.dot(input_vector, course_vector) / (np.linalg.norm(input_vector) * np.linalg.norm(course_vector))

      # Store similarity score
      similarities[course_id] = similarity_score

    # Sort courses by similarity score in descending order
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

    # Return most similar courses and their similarity scores
    return [
      {
        "course_id":               course_id,
        "course_title":            self.course_data[course_id]["title"],
        "course_description":      self.course_data[course_id]["description"],
        "course_similarity_score": similarity_score,
      }
      for (course_id, similarity_score) in sorted_similarities
    ]
  
  
def main():
  # Parse command line arguments
  parser = ArgumentParser()
  parser.add_argument("id",   type=str, help="Course ID to find similar courses for")
  parser.add_argument("data", type=str, help="Path to course data JSON file"        )
  parser.add_argument("n",    type=int, help="Number of similar courses to find"    )
  args = parser.parse_args()

  # Build vector database with course data
  with ScopeTimer("Building vector database"):
    vector_database = CourseSimilarityVectorDatabase(args.data)
  
  # Find N most similar courses
  n_most_similar_courses = vector_database.find_most_similar_courses(args.id)[:args.n]

  #
  # Print results
  #

  for (index, course_info) in enumerate(n_most_similar_courses):
    course_id               = course_info["course_id"]
    course_title            = course_info["course_title"]
    course_description      = course_info["course_description"]
    course_similarity_score = course_info["course_similarity_score"]
    print(col(f"({index}) Similar Course:", "green"), col(course_id, "light_green"))
    print(col("Title:", "green"), col(f"{course_title}", "light_green"))
    print(col("Description:", "green"), col(course_description, "light_green"))
    print(col("Similarity Score:", "green"), col(f"{course_similarity_score:.2f}", "light_green"))
    print()


if __name__ == "__main__":
  main()

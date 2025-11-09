import numpy as np

np.random.seed(30)

arr = np.random.randint(40, 101, (10, 5))
print("Исходные оценки:\n", arr)

student_mean = np.mean(arr, axis=1)
print("Средний балл студентов:", student_mean)

subject_mean = np.mean(arr, axis=0)
print("Средний балл по предметам:", subject_mean)

print("Общий средний балл:", np.mean(arr))

best_student = np.argmax(student_mean)
hardest_subject = np.argmin(subject_mean)
print(f"Лучший студент: №{best_student + 1}")
print(f"Самый трудный предмет: №{hardest_subject + 1}")

arr = np.clip(arr + 5, 0, 100)
print("После корректировки (+5 баллов каждому):\n", arr)

sorted_indices = np.argsort(student_mean)[::-1]
print("Студенты по убыванию среднего балла:\n", arr[sorted_indices])
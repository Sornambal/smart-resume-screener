def match_score(cleaned_text, skill_list):
    score = 0
    found_skills = []
    for skill in skill_list:
        if skill.lower() in cleaned_text:
            score += 1
            found_skills.append(skill)
    return score, found_skills

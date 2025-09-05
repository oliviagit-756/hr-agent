# matcher/api.py
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .services import (
    health_info, search_candidates_for_jd, search_jobs_for_resume, pick_jd_text_by_keywords
)

@api_view(["GET"])
def health(_request):
    return Response(health_info())

@api_view(["POST"])
def search_candidates(request):
    """
    JD -> top resumes
    Body: { "jd_text": "...", "keywords": "...", "k": 25, "top_m": 10 }
    Provide EITHER jd_text OR keywords.
    """
    jd_text = request.data.get("jd_text")
    keywords = request.data.get("keywords")
    k = int(request.data.get("k", 25))
    top_m = int(request.data.get("top_m", 10))

    if not jd_text:
        if not keywords:
            return Response({"detail": "Provide 'jd_text' or 'keywords'."}, status=status.HTTP_400_BAD_REQUEST)
        hit = pick_jd_text_by_keywords(keywords)
        if not hit:
            return Response({"detail": f"No JD matched keywords: {keywords}"}, status=status.HTTP_404_NOT_FOUND)
        jd_text = hit["text"]

    try:
        df = search_candidates_for_jd(jd_text, k=k, top_m=top_m)
    except Exception as e:
        return Response({"detail": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    return Response(df.to_dict(orient="records"))

@api_view(["POST"])
def search_jobs(request):
    """
    Resume -> top JDs
    Body: { "resume_text": "...", "k": 25, "top_m": 10 }
    """
    resume_text = request.data.get("resume_text")
    if not resume_text:
        return Response({"detail": "Missing 'resume_text'."}, status=status.HTTP_400_BAD_REQUEST)
    k = int(request.data.get("k", 25))
    top_m = int(request.data.get("top_m", 10))
    try:
        df = search_jobs_for_resume(resume_text, k=k, top_m=top_m)
    except Exception as e:
        return Response({"detail": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    return Response(df.to_dict(orient="records"))

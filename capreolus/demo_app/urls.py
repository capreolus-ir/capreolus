from django.urls import path

from capreolus.demo_app import views

urlpatterns = [
    path("", views.ConfigsView.as_view(), name="index"),
    path("experiment/", views.ExperimentView.as_view(), name="experiment"),
    path("query/", views.QueryView.as_view(), name="query"),
    path("query_suggestion/", views.QuerySuggestionView.as_view(), name="query_suggestion"),
    path("document/", views.DocumentView.as_view(), name="document"),
    path("compare/", views.CompareExperimentsView.as_view(), name="compare"),
    path("compare_query/", views.CompareQueryView.as_view(), name="compare_query"),
]

// src/app/services/chatbot.service.ts
import { Injectable } from '@angular/core';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { catchError, tap } from 'rxjs/operators';

// Interface pour la réponse simplifiée
export interface SimpleChatResponse {
  question: string;
  answer: string;
  context_provided_to_llm?: string;
  sources?: any[];
  mode?: string;
  error?: string;
}

@Injectable({
  providedIn: 'root'
})
export class ChatbotService { // <--- Nom de la classe mis à jour
  private apiUrl = 'http://localhost:5001/api';

  constructor(private http: HttpClient) { }

  sendMessage(question: string): Observable<SimpleChatResponse> {
    return this.http.post<SimpleChatResponse>(`${this.apiUrl}/chat`, { question }).pipe(
      tap(response => console.log('API Chat Response:', response)),
      catchError(this.handleError)
    );
  }

  getStatus(): Observable<any> {
    return this.http.get<any>(`${this.apiUrl}/status`).pipe(
      tap(status => console.log('API Status:', status)),
      catchError(this.handleError)
    );
  }

  private handleError(error: HttpErrorResponse) {
    let errorMessage = 'Une erreur inconnue est survenue !';
    if (error.error instanceof ErrorEvent) {
      errorMessage = `Erreur : ${error.error.message}`;
    } else {
      errorMessage = `Erreur du serveur : ${error.status} - ${error.error?.error || error.statusText}`;
      console.error('Backend error details:', error.error);
    }
    return throwError(() => new Error(errorMessage));
  }
}
// src/app/components/chatbot/chatbot.component.ts
import { Component, OnInit } from '@angular/core';
// Ajustez le chemin si votre service est dans un sous-dossier de 'services'
import { ChatbotService, SimpleChatResponse } from '../services/chatbot.service';

@Component({
  selector: 'app-chatbot', // Le sélecteur est généralement 'app-' + nom du composant
  standalone:false,
  templateUrl: './chatbot.component.html', // <--- Mis à jour
  styleUrls: ['./chatbot.component.css']   // <--- Mis à jour
})
export class ChatbotComponent implements OnInit { // <--- Nom de la classe mis à jour
  userQuestion: string = '';
  botResponse: SimpleChatResponse | null = null;
  isLoading: boolean = false;
  errorMessage: string | null = null;
  backendStatus: any = null;

  constructor(private chatbotService: ChatbotService) { } // <--- Service injecté mis à jour

  ngOnInit(): void {
    this.chatbotService.getStatus().subscribe({ // <--- Service utilisé mis à jour
      next: (status) => this.backendStatus = status,
      error: (err) => this.errorMessage = `Erreur statut: ${err.message}`
    });
  }

  askQuestion(): void {
    if (!this.userQuestion.trim()) {
      return;
    }
    this.isLoading = true;
    this.botResponse = null;
    this.errorMessage = null;

    this.chatbotService.sendMessage(this.userQuestion).subscribe({ // <--- Service utilisé mis à jour
      next: (response) => {
        this.botResponse = response;
        this.isLoading = false;
        this.userQuestion = '';
      },
      error: (err) => {
        this.errorMessage = err.message;
        this.isLoading = false;
      }
    });
  }
}
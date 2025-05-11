import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { HttpClientModule } from '@angular/common/http';
import { FormsModule } from '@angular/forms';
import { AppComponent } from './app.component';
// Ajustez le chemin si votre composant est dans un sous-dossier de 'components'
import { ChatbotComponent } from './chatbot/chatbot.component'; // <--- Mis à jour

@NgModule({
  declarations: [
    AppComponent,
    ChatbotComponent // <--- Mis à jour
  ],
  imports: [
    BrowserModule,
    HttpClientModule,
    FormsModule
  ],
  providers: [
    // ChatbotService est déjà providedIn: 'root'
  ],
  bootstrap: [AppComponent]
})
export class AppModule { }
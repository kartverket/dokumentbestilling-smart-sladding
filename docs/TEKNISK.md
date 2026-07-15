# Smart sladding: Teknisk beskrivelse

Løsningen for smart sladding består av fire deler: en batch-jobb som styrer produksjonsflyten, et modell-API som utfører selve analysen, et sett analyse- og testverktøy, og en treningspipeline for bildemodellen. Figur 1 viser hvordan delene henger sammen.

I produksjon henter batch-jobben ubehandlede dokumenter fra databasen og kontrollerer at de fortsatt er klare for behandling. For hvert dokument lastes PDF-en ned fra dokument-APIet og sendes til modell-APIet, som returnerer koordinater for områdene som skal sladdes. Forslagene lagres deretter i databasen, merket som maskingenererte slik at de kan skilles fra manuelle sladdinger og eventuelt overprøves. Batch-jobben inneholder ingen maskinlæring selv, den er et lettvekts bindeledd mellom systemene, og alt tungt arbeid skjer i modell-APIet.

Utenfor produksjonsløpet finnes to støtteprosesser. Analyse- og testverktøyene kjører den samme modellkoden lokalt mot testdokumenter med kjent fasit, og måler hvor stor andel av fasit-sladdingene modellen finner. Treningspipelinen bruker labels fra eksisterende løsning som treningsdata og produserer nye vekter til bildemodellen, som så tas i bruk av modell-APIet.

![Figur 1: Overordnet arkitektur](diagrams/skjermbilder/arkitektur.png)

**Figur 1:** Overordnet arkitektur. Batch-jobben henter ubehandlede dokumenter fra databasen (1), laster ned PDF-en fra dokument-APIet (2) og sender den til modell-APIet, som returnerer sladde-bokser (3). Forslagene lagres i databasen som maskingenererte sladdinger (4). Analyse- og testverktøyene og treningspipelinen inngår ikke i produksjonsflyten, men brukes til å måle og forbedre modellen.

---

Selve deteksjonen bygger på to uavhengige spor som utfyller hverandre, vist i figur 2. Først gjøres hver side i PDF-en om til et bilde. Deretter leter to metoder etter fødselsnummer parallelt: Et tekstspor der OCR leser teksten på siden og validerer at tallrekker faktisk er gyldige fødselsnummer, og et bildespor der en trent bildemodell (YOLO) ser etter områder som visuelt ligner et fødselsnummer, også der teksten er vanskelig å lese maskinelt, for eksempel ved håndskrift, stempler eller dårlig skannekvalitet.

Tekstsporet gir høy presisjon fordi hvert funn kan kontrolleres matematisk, mens bildesporet gir dekning i tilfellene OCR-en ikke klarer å lese. Til slutt slås funnene fra de to sporene sammen og kvalitetssjekkes, før det legges sladdebokser over fødselsnumrene og koordinatene returneres som JSON til den som kalte tjenesten.

![Figur 2: Overordnet flyt](diagrams/skjermbilder/flyt.png)

**Figur 2:** Overordnet flyt for sladding av ett dokument. To uavhengige spor, tekstgjenkjenning (OCR) og bildegjenkjenning (YOLO), leter etter fødselsnummer parallelt. Funnene slås sammen og kvalitetssjekkes før sladdeboksene returneres.

---

Figur 3 viser pipelinen i detalj, med kriteriene som avgjør om et funn godtas. Hver side rendres til bilde i 300 DPI. En orienteringsmodell klassifiserer om siden ligger riktig vei (0, 90, 180 eller 270 grader), og siden roteres bare dersom modellen er tilstrekkelig sikker (konfidens ≥ 0,7), ellers beholdes originalorienteringen.

I tekstsporet leser OCR-motoren (PaddleOCR) teksten på siden som enkeltord med posisjon. Ordene grupperes til linjer, og vanlige OCR-forvekslinger normaliseres før analysen (for eksempel tolkes bokstaven O som tallet 0 og l som 1). Et glidende vindu ser deretter på elleve og elleve siffer om gangen. En kandidat godtas bare hvis tre krav er oppfylt samtidig: Sifrene henger tilnærmet sammen (maks tre avbrudd på inntil to tegn, og kun mellomrom eller enkel tegnsetting), de seks første sifrene utgjør en gyldig dato (inkludert d-nummer, der dagen er tillagt 40), og begge kontrollsifrene stemmer etter modulus 11-kontrollen. Kombinasjonen gjør at tilfeldige tallrekker, telefonnummer og beløp svært sjelden slipper gjennom.

I bildesporet foreslår YOLO-modellen områder som ligner fødselsnummer, med en konfidensverdi per funn. Forslagene holdes så opp mot tekstsporet: Overlapper et YOLO-funn i hovedsak (mer enn 50 %) med et OCR-validert funn, regnes det som samme treff og merkes med kilde «begge». Står YOLO-funnet alene, avhenger kravet av konteksten. Har OCR-en lest tekst i området, gjøres en enkel innholdssjekk, området må inneholde minst ett siffer og maks én bokstav, siden to eller flere bokstaver utelukker et fødselsnummer uansett hvor sikker bildemodellen er. Har OCR-en ikke lest noe i området, kreves i stedet en konfidens på minst 0,40. For vertikale områder (stående tekst), som OCR-en normalt ikke leser, kreves en konfidens på minst 0,90.

For hvert godkjente funn beregnes en sladdeboks som dekker de fem siste sifrene, personnummerdelen, slik at fødselsdatoen forblir leselig mens den identifiserende delen skjules. Ble siden rotert før analysen, regnes koordinatene til slutt tilbake til dokumentets opprinnelige orientering, og resultatet returneres som JSON med posisjon, kilde og eventuell konfidensverdi per boks.

![Figur 3: Deteksjonspipelinen i detalj](diagrams/skjermbilder/pipeline.png)

**Figur 3:** Deteksjonspipelinen i detalj. Tekstsporet validerer kandidater med krav til sammenheng, datoform og modulus 11-kontroll. Bildesporet (YOLO) bidrar med funn som kvalitetssjekkes ulikt avhengig av om OCR-en har lest tekst i området: innholdssjekk der tekst finnes, konfidenskrav på 0,40 uten tekst, og 0,90 for vertikale områder. Sladdeboksen legges over de fem siste sifrene i nummeret.

---

## Videre forbedringer

Løsningen har muligheter for videre forbedring som ikke ble utnyttet i dette arbeidet, hovedsakelig fordi de krever mer GPU-minne enn serveren som var tilgjengelig hadde.

Den enkleste gevinsten er ytelse. OCR-motoren støtter høyytelses-inferens (HPI), som allerede ligger klargjort i koden, men avslått fordi det krever et kraftigere skjermkort. Aktiveres dette på en større GPU, vil behandlingstiden per dokument gå ned. Av samme grunn ble flere konfigurasjonsparametere holdt lavere enn ønskelig, blant annet oppløsningen inn til tekstdeteksjonen og bildemodellen samt antall sider og tekstlinjer som behandles per batch. Siden alt dette er samlet som konfigurasjonsparametere, kan det skrus opp på kraftigere maskinvare uten endringer i selve koden.

---
name: jira-sak
description: Skriv Jira-saker for team Eiet i teamets faste mal (Brukerhistorie, Løsningsbeskrivelse, Automatiske tester, Akseptansekriterier, Sikkerhet- og personvernvurderinger). Bruk når noen ber om en ny Jira-sak, en brukerhistorie, en oppfølgingssak eller vil gjøre om et notat, en feil eller et funn til en sak.
---

# Jira-saker for team Eiet

Alle saker skrives på norsk bokmål i malen under. Seksjonene, rekkefølgen,
emojiene og overskriftsnivåene er faste. Teksten leveres som Markdown, som
Jira gjør om til riktig formatering når den limes inn i beskrivelsesfeltet.

## Ordbruk

* Skriv aldri «saksbehandler». Den som behandler dokumentene heter
  **behandler**, også i sammensetninger («behandleren», «behandlerens»).
* Bruk de samme ordene som løsningen viser behandleren: «sladd»,
  «sladdeforslag», «godkjenne», «avvise».
* Står en enumverdi eller et feltnavn i koden (`ACCEPTED`, `ml_status`), skriv
  det i kodeformat, slik at det ikke leses som prosa.

## Malen

Kopier strukturen nøyaktig. Bare teksten i vinkelparentes byttes ut.

```markdown
## 🧍 Brukerhistorie

**Som en** <hvem>
**Ønsker jeg** <hva>
**Slik at** <hvorfor>

## 🎯 Løsningsbeskrivelse

<Forklaring/oppsummering>

#### 🧪 Automatiske tester

Ved utvikling av saken skal det samtidig opprettes e2e-tester.

## ⚙️ Akseptansekriterier

* Gitt at <noe skjer>, så skal
  * <Jeg ha denne muligheten>
    * <Med dette valget>
  * <og denne muligheten>
* Gitt at <noe annet skjer>, så skal
  * <…>

## 🦺 Sikkerhet- og personvernvurderinger

Ved videreutvikling må vi vurdere om endringen påvirker gjeldende sikkerhet- og personvernsvurderinger.

* Endrer saken (IP/DPIA) personvernvurderingene? Ja/nei (kort begrunnelse)
* Endrer saken (ROS) sikkerhetsvurderingene? Ja/nei (kort begrunnelse)
```

## Slik fyller du ut seksjonene

**Brukerhistorie.** Tre linjer, med «Som en», «Ønsker jeg» og «Slik at» i
fet skrift. «Hvem» er en rolle, for eksempel «behandler» eller «en i team
Eiet». «Slik at» sier hvilken nytte rollen får, ikke hvordan det løses.

**Løsningsbeskrivelse.** Si først hva som skjer i dag, så hva som skal skje
etterpå. Er årsaken ukjent, si det rett ut og beskriv hva som må undersøkes
først. Ikke skriv at noe er årsaken før det er bekreftet. Er saken en
oppfølging av en annen sak, lenk den i første avsnitt. Hold den kort: det
som hører hjemme i akseptansekriteriene, skal stå der.

**Automatiske tester.** Setningen står alltid uendret. Den er ikke valgfri.

**Akseptansekriterier.** Hvert kriterium starter med «Gitt at …, så skal»,
med det som skal skje som underpunkter. Et kriterium skal kunne testes: en
annen utvikler skal kunne avgjøre om det er oppfylt uten å spørre. Ta med
kanttilfellene som kan gå galt, og påvirkes statistikk, eksport eller
treningsdata, skal det stå et kriterium for det også.

**Sikkerhet- og personvernvurderinger.** Svar alltid Ja eller Nei på begge
spørsmålene, med én setning om hvorfor. «Nei» krever også en begrunnelse,
for eksempel «Saken endrer ikke hvilke persondata vi lagrer eller hvem som
ser dem». Berører saken fødselsnummer, sladding eller tilgang, vurder
spørsmålet på ordentlig i stedet for å svare nei av vane.

## Språk

Korte setninger i aktiv form. Ingen tankestreker, ingen fete stikkord foran
kulepunkter utover dem malen har, og ingen avsluttende oppsummering.

## Tittel

Tittelen er en kort beskrivelse av endringen, ikke av problemet:
«Gi flyttede sladdeforslag egen status», ikke «Feil status ved flytting».
Lever tittelen på egen linje over beskrivelsen.

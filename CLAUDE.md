# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Meeko is an intelligent voice assistant. It is developed on Mac, and will eventually be available on Raspberry Pi. It uses Picovoice for local wake word detection, Deepgram for STT/TTS, and Claude Sonnet 4.6 as the LLM.

The project is structured into 7 milestones, each producing a testable system. Milestones 1-5 run on Mac (real mic + speakers). Hardware is purchased in Milestone 6.


## Architecture

Two possible stacks — the choice is made at Milestone 4:

**Voice Agent stack (Milestones 1-3):** Single Deepgram Voice Agent WebSocket handles STT + LLM + TTS orchestration. Simpler but ~$68/month.

**DIY stack (Milestone 4+):** Direct Deepgram STT -> Claude API -> Deepgram TTS calls. More control, ~$31/month, requires barge-in implementation.

Core loop: Wake word -> stream mic to Deepgram STT -> end-of-turn detected -> transcript to Claude with conversation history -> stream response sentences to Deepgram TTS -> play audio -> return to wake word listening.


# BlenderGen Pipeline

BlenderGen Pipeline is an end-to-end application for generating 3D models from concepts or images, rendering them in Blender, and producing visual outputs (GIFs/videos). This document describes the current implementation of the UI with all its features and functionalities.

## Overview

The application follows a non-linear pipeline approach, allowing users to:
- Generate or upload concept images
- Create 3D models from selected images
- Render the models in Blender
- Generate output animations (GIF/MP4)

The UI is designed to be intuitive, visually appealing, and provide clear feedback at each stage of the process.

## Features

### 1. Concept & Images Stage

**Input Methods:**
- **Text-to-Image**: Enter a concept or prompt and generate multiple images
- **Image Upload**: Upload existing images for 3D model generation

**Functionality:**
- AI refinement toggle for text prompts
- Real-time progress tracking during image generation
- Image gallery with selection mechanism
- Support for multiple image formats (PNG, JPG, WEBP)

### 2. 3D Model Generation Stage

**Model Types:**
- Tripo
- Trellis
- Hunyuan

**Functionality:**
- Model type selection via radio buttons
- Real-time progress tracking during model generation
- Error handling for missing prerequisites (no image selected)
- Visual representation of the generated 3D model
- File path display for the generated model

### 3. Rendering & Output Stage

**Rendering Options:**
- Interactive mode toggle

**Output Types:**
- GIF animation
- MP4 video

**Functionality:**
- Real-time progress tracking during rendering
- Error handling for missing prerequisites (no model generated)
- Visual representation of the rendering process
- File path display for the generated outputs
- Option to re-render with current settings

### Global Features

**Progress Tracking:**
- Expandable side panel for process logs
- Real-time progress indicators
- Detailed logging of operations
- Status message display

**File Management:**
- Automatic file organization with unique project IDs
- Structured directory hierarchy for different artifact types:
  - `refs/`: Reference images and concepts
  - `export/`: Generated 3D models
  - `renders/`: Rendered outputs (GIF/MP4)
  - `scenes/`: Blender scene files

**UI Components:**
- Consistent blue progress bars across all stages
- Modern image placeholders with appropriate icons
- Responsive layout that adapts to different screen sizes
- Clear visual separation between pipeline stages
- Consistent styling with a dark theme optimized for visual content

## Non-Linear Workflow

Unlike traditional linear pipelines, BlenderGen Pipeline supports a flexible, non-linear workflow:

- Users can change concept selections at any time
- New images can be uploaded even after model generation
- New models can be generated from different images
- Re-rendering is possible with different settings
- Each stage maintains its state independently

This approach allows for creative exploration and iteration without starting over.

## Technical Implementation

The application is built with:
- Next.js App Router for routing and server components
- React for the UI components
- Tailwind CSS for styling
- TypeScript for type safety
- Shadcn UI components for consistent design

File operations use a structured approach with:
- Unique project IDs generated with UUID
- Timestamped filenames for versioning
- Consistent file path utilities

## Getting Started

1. Clone the repository
2. Install dependencies with `npm install`
3. Run the development server with `npm run dev`
4. Open [http://localhost:3000](http://localhost:3000) in your browser

## Future Enhancements

Planned enhancements include:
- 3D model preview with three.js
- Additional model generation options
- More rendering customization options
- Project saving and loading
- Batch processing capabilities


# BlenderGen Pipeline UI Design Document

## Overview

BlenderGen Pipeline is an end-to-end application that generates 3D models from concepts or images, renders them in Blender, and produces visual outputs (GIFs/videos). Instead of using tabs as in the current implementation, this design document describes a continuous scrolling pipeline where each stage flows naturally into the next.

## Core Principles

1. **Continuous Flow**: Users should be guided through a logical, step-by-step process
2. **Visual Feedback**: Each step should provide clear visual indicators of progress and results
3. **Flexibility**: Allow users to enter the pipeline at different points (concept, image, or 3D model)
4. **Progressive Disclosure**: Show only relevant controls for the current stage of the process

## Pipeline Stages

The UI is organized into three main pipeline stages, each with its own section in the scrollable interface:

### 1. Concept & Images Stage

**Purpose**: Generate or select the 2D image that will be used for 3D model creation.

**Input Options**:
- **Text-to-Image Mode**: Enter a concept or prompt, refine it with AI, and generate multiple images
- **Direct Image Upload**: Use an existing image which is automatically processed for 3D generation

**Key Components**:
- Segmented control to switch between text-to-image and image upload workflows
- Text area for concept/prompt input 
- AI refinement toggle (when in text-to-image mode)
- Image browsing and preview section (when in image upload mode)
- Generated/selected images gallery with selection mechanism
- "Proceed to 3D Model" action button

**Flow Behavior**:
- Initially, only the input methods are shown
- Once an image is generated or uploaded, the image gallery appears below
- When an image is selected, the "Proceed to 3D Model" button appears
- Clicking this button auto-scrolls to the 3D Model section

### 2. 3D Model Generation Stage

**Purpose**: Transform the selected 2D image into a 3D model, with options for different generation techniques.

**Key Components**:
- Selected image display (reminder of what is being processed)
- Model type selection (Tripo, Trellis, Hunyuan, or try all)
- Face limit/resolution control
- "Generate 3D Model" action button
- 3D model result display with information about the generated file
- Preview capability (QuickLook on macOS)
- "Proceed to Rendering" action button

**Flow Behavior**:
- This section becomes active when an image has been selected
- When a 3D model is generated, the preview and "Proceed to Rendering" options appear
- The generated model path is displayed along with statistics
- Clicking "Proceed to Rendering" auto-scrolls to the Output section

### 3. Rendering & Output Stage

**Purpose**: Render the 3D model in Blender and create output animations (GIF/MP4).

**Key Components**:
- Blender rendering settings:
  - Material selection dropdown
  - Model height input
  - Texture usage toggle
  - Interactive mode toggle
- "Run Blender Rendering" action button
- Render results display with folder path
- Output creation controls:
  - GIF options (duration, resize factor)
  - "Create GIF" and "Create MP4" action buttons
- Output results display with links to generated files

**Flow Behavior**:
- This section becomes active when a 3D model has been generated
- After Blender rendering is complete, the output creation controls become available
- Generated outputs are displayed with their file paths and previews where possible

## Global UI Elements

### Progress Tracking

A fixed progress area at the bottom of the application provides:
- Text log of operations and progress
- Status message display
- Progress bar for longer operations

### Continuous Scrolling Implementation

Instead of tabs, the UI uses a single scrollable container with:
- Clear visual separation between pipeline stages
- Auto-scrolling to the next relevant section when a stage is completed
- Animated transitions between stages for visual continuity
- Collapsible previous stages to maintain focus on the current task

### Navigation Controls

- Section headers with collapsible content
- "Jump to" links for quick navigation between sections
- Visual indicators of the current active stage

## Responsive Design Guidelines

- Use relative sizing and flexible layouts
- Maintain minimum functional width of 800px
- Support dynamic height adjustment for various screen sizes
- Use scrolling within sections when needed instead of overflowing

## Error Handling

- Inline error indicators at the point of failure
- Error recovery suggestions
- Ability to retry failed operations
- Detailed logs accessible via an expandable panel

## Visual Design Language

- Clean, modern interface with consistent padding and spacing
- Primary colors for action buttons, neutral colors for containers
- Clear visual hierarchy with section headers
- Visual progress indicators between pipeline stages
- High-contrast text for readability

## Special Features

### 3D Model Preview

- Integrated 3D model viewer if available in the framework
- Fallback to native system viewer (QuickLook on macOS)
- Thumbnail generation for quick identification

### Asynchronous Processing

All potentially time-consuming operations run asynchronously with:
- Cancellation capabilities for long-running processes
- Background processing with visual feedback
- Thread management to prevent UI freezing

## Implementation Considerations

- Use a reactive framework to maintain state consistency
- Implement virtualized lists for large image galleries
- Use efficient image loading and caching for performance
- Consider WebGL for 3D model preview if the framework supports it
- Implement graceful fallbacks for platform-specific features

## User Interaction Flow

1. User enters at the top of the pipeline (concept or image)
2. As each stage completes, the UI scrolls to reveal the next stage
3. Previous stages remain accessible but collapse to summaries
4. Output files are presented with preview and file location information
5. Users can save their entire pipeline configuration for future reuse

This design provides a guided experience while maintaining flexibility for different user workflows and entry points. 